local LrSocket = import 'LrSocket'
local LrTasks = import 'LrTasks'
local LrFunctionContext = import 'LrFunctionContext'
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrLogger = import 'LrLogger'
local LrPathUtils = import 'LrPathUtils'
local LrFileUtils = import 'LrFileUtils'
local LrExportSession = import 'LrExportSession'

-- Create and enable logger
local logger = LrLogger('XMPPlayer')
logger:enable('logfile')
local LOG_VERBOSE = false

local function logInfo(message)
    if LOG_VERBOSE then
        logger:info(message)
    end
end

-- Global variable to track server status
local SERVER_STATUS = {
    running = false,
    port = 7878,
    error = nil
}

local CATALOG_LRU_LIMIT = 100
local importedPhotoLru = {}
local importedPhotoPaths = {}
local catalogLruSeeded = false
local HEALTH_FILE = "/tmp/lightroom_bridge_health.txt"

local function getCatalogLimit()
    return CATALOG_LRU_LIMIT
end

local function touchImportedPhoto(photoPath)
    if not photoPath or photoPath == "" then
        return
    end

    if importedPhotoPaths[photoPath] then
        for i = #importedPhotoLru, 1, -1 do
            if importedPhotoLru[i] == photoPath then
                table.remove(importedPhotoLru, i)
                break
            end
        end
    end

    table.insert(importedPhotoLru, photoPath)
    importedPhotoPaths[photoPath] = true
end

local function seedCatalogLru(catalog)
    if catalogLruSeeded then
        return
    end

    catalogLruSeeded = true
    local ok, photosOrErr = LrTasks.pcall(function()
        return catalog:getAllPhotos()
    end)
    if not ok or type(photosOrErr) ~= "table" then
        logger:error("Failed to seed catalog LRU from existing catalog: " .. tostring(photosOrErr))
        return
    end

    for _, photo in ipairs(photosOrErr) do
        local pathOk, photoPath = LrTasks.pcall(function()
            return photo:getRawMetadata("path")
        end)
        if pathOk and photoPath and string.find(photoPath, "lightroom_task_", 1, true) then
            touchImportedPhoto(photoPath)
        end
    end

    logInfo("Seeded catalog LRU with " .. tostring(#importedPhotoLru) .. " existing task photos")
end

local function evictCatalogLru(catalog, keepPaths)
    local limit = getCatalogLimit()
    local removedCount = 0
    keepPaths = keepPaths or {}
    seedCatalogLru(catalog)
    for keepPath, _ in pairs(keepPaths) do
        touchImportedPhoto(keepPath)
    end

    while #importedPhotoLru > limit do
        local oldPath = table.remove(importedPhotoLru, 1)
        importedPhotoPaths[oldPath] = nil

        if oldPath and not keepPaths[oldPath] then
            local photo = catalog:findPhotoByPath(oldPath)
            if photo then
                local ok, removeErr = LrTasks.pcall(function()
                    catalog:removePhotos({ photo })
                end)
                if ok then
                    removedCount = removedCount + 1
                    logInfo("Removed old LRU photo from catalog: " .. oldPath)
                else
                    logger:error("Failed to remove old LRU photo from catalog: " .. oldPath .. " error=" .. tostring(removeErr))
                end
            end
        end
    end

    return removedCount
end

local function getCatalogLruStatus()
    return "catalog_lru_limit=" .. tostring(getCatalogLimit()) .. ";catalog_lru_size=" .. tostring(#importedPhotoLru)
end

local function writeHealthFile()
    local file = io.open(HEALTH_FILE, "w")
    if not file then
        return
    end
    file:write(getCatalogLruStatus())
    file:write(";updated_at=" .. tostring(os.time()))
    file:close()
end

local function sanitizeStatusValue(value)
    if value == nil then
        return ""
    end
    local text = tostring(value)
    text = string.gsub(text, "[\r\n;]", " ")
    return text
end

local function writeRenderStatus(outputDir, status, fields)
    if not outputDir or outputDir == "" then
        return
    end
    local statusPath = LrPathUtils.child(outputDir, ".lightroom_render_status")
    if not statusPath then
        return
    end
    local file = io.open(statusPath, "w")
    if not file then
        return
    end
    file:write("status=" .. sanitizeStatusValue(status))
    file:write(";updated_at=" .. tostring(os.time()))
    if type(fields) == "table" then
        for key, value in pairs(fields) do
            file:write(";" .. sanitizeStatusValue(key) .. "=" .. sanitizeStatusValue(value))
        end
    end
    file:close()
end

local function failRender(outputDir, taskId, code, message, retryable)
    logger:error(message)
    writeRenderStatus(outputDir, "error", {
        task_id = taskId or "",
        error_code = code,
        message = message,
        retryable = retryable and "true" or "false",
    })
end

local function sanitizeFileStem(value)
    if value == nil or value == "" then
        return nil
    end
    local text = tostring(value)
    text = string.gsub(text, "[^%w%._%-]", "_")
    text = string.gsub(text, "^[%._%-]+", "")
    text = string.gsub(text, "[%._%-]+$", "")
    if text == "" then
        return nil
    end
    if string.len(text) > 180 then
        text = string.sub(text, 1, 180)
    end
    return text
end



-- Add path processing functions
local function getPathParts(path)
    local parts = {}
    local current = path
    while current and current ~= "" do
        local name = LrPathUtils.leafName(current)
        if name then
            table.insert(parts, 1, name)
        end
        current = LrPathUtils.parent(current)
    end
    return parts
end

local function getParentDir(path)
    return LrPathUtils.parent(path) or "."
end

local function splitMessage(message)
    local result = {}
    local pattern = "[^|]+"
    local start = 1
    local splitStart, splitEnd = string.find(message, pattern, start)
    
    while splitStart do
        table.insert(result, string.sub(message, splitStart, splitEnd))
        start = splitEnd + 2  -- +2 to skip the delimiter
        splitStart, splitEnd = string.find(message, pattern, start)
    end
    
    return result
end

local function trimString(value)
    if not value then
        return value
    end
    return string.gsub(value, "^%s*(.-)%s*$", "%1")
end

-- Add Lua settings file parsing function
local function parseLuaSettingsFile(luaPath)
    logInfo("Loading settings from Lua file: " .. luaPath)
    
    -- Use loadfile to load Lua file
    local chunk, err = loadfile(luaPath)
    if not chunk then
        logger:error("Failed to load Lua file: " .. tostring(err))
        return nil
    end
    
    -- Execute file and get return value
    local success, settings = pcall(chunk)
    if not success then
        logger:error("Failed to execute Lua file: " .. tostring(settings))
        return nil
    end
    
    -- Ensure return value is a table
    if type(settings) ~= "table" then
        logger:error("Lua file did not return a table")
        return nil
    end
    
    logInfo("Successfully loaded settings from Lua file")
    return settings
end

-- Modified importPreset function
local function importPreset(settingsPath)
    -- Get filename (as preset name)
    local presetName = LrPathUtils.removeExtension(LrPathUtils.leafName(settingsPath))
    if not presetName then
        logger:error("Failed to get preset name from path: " .. settingsPath)
        return nil
    end
    
    -- Parse settings file
    local settings = parseLuaSettingsFile(settingsPath)
    if not settings then
        logger:error("Failed to parse settings file")
        return nil
    end    
    return settings
end

local function tableToString(t, indent)
    if not t then return "nil" end
    
    local result = {}
    indent = indent or ""
    
    for k, v in pairs(t) do
        if type(v) == "table" then
            table.insert(result, indent .. k .. ":\n" .. tableToString(v, indent .. "  "))
        else
            table.insert(result, indent .. k .. " = " .. tostring(v))
        end
    end
    
    return table.concat(result, "\n")
end

local function tableContainsMaskSettings(t)
    if type(t) ~= "table" then
        return false
    end

    for k, v in pairs(t) do
        local key = tostring(k)
        if string.find(key, "Mask") or key == "LocalizedCorrections" or key == "RetouchAreas" then
            return true
        end
        if type(v) == "table" and tableContainsMaskSettings(v) then
            return true
        end
    end

    return false
end

local function handleRequest(message)
    logInfo("Processing request: " .. tostring(message))
    
    -- Check if message is empty
    if not message or message == "" then
        return "error|Empty message received"
    end
    
    -- Try to parse request
    local parts = splitMessage(message)
    
    if #parts == 0 then
        return "error|Empty request"
    end
    
    local command = trimString(parts[1])
    logInfo("Processing command: " .. command)
    
    -- Handle ping request
    if command == "ping" then
        return "pong"
    end

    if command == "health" then
        writeHealthFile()
        return "health|" .. getCatalogLruStatus()
    end
    
    -- Handle photo request
    if command == "process" and parts[2] and parts[3] then
        local photoPath = parts[2]
        local xmpPath = parts[3]
        local requestedOutputDir = parts[4]
        local taskId = parts[5] or ""
        local requestedOutputStem = parts[6]
        
        -- Validate paths and file existence
        if not photoPath or photoPath == "" then
            return "error|Invalid photo path"
        end
        
        if not xmpPath or xmpPath == "" then
            return "error|Invalid XMP path"
        end
        
        if not LrFileUtils.exists(photoPath) then
            return "error|Photo file does not exist"
        end
        
        if not LrFileUtils.exists(xmpPath) then
            return "error|XMP file does not exist"
        end
        
        -- Prepare output path. Newer clients pass a per-task output directory so
        -- concurrent tasks do not have to guess Lightroom's export destination.
        local outputDir = requestedOutputDir
        if not outputDir or outputDir == "" then
            outputDir = LrPathUtils.child(getParentDir(photoPath), "processed")
        end
        if not outputDir then
            return "error|Failed to get parent directory from path: " .. photoPath
        end
        
        local fileName = LrPathUtils.leafName(photoPath)
        local photoName = sanitizeFileStem(requestedOutputStem) or LrPathUtils.removeExtension(fileName)
        if not photoName then
            return "error|Failed to get photo name from path: " .. photoPath
        end
        
        local ok, mkdirErr = pcall(function()
            LrFileUtils.createAllDirectories(outputDir)
        end)
        if not ok then
            logger:error("Failed to create output directory: " .. tostring(mkdirErr))
            return "error|Failed to create output directory"
        end

        local outputPath = LrPathUtils.child(outputDir, photoName .. ".jpg")
        if not outputPath then
            return "error|Failed to create output path"
        end

        writeRenderStatus(outputDir, "queued", {
            task_id = taskId,
            output_path = outputPath,
        })
        
        -- Process photo in new async task.
        -- Wrap the whole body in LrTasks.pcall so any failure (apply settings,
        -- updateAISettings, export) is logged instead of surfacing as an
        -- uncaught Lua error / Lightroom error dialog.
        LrTasks.startAsyncTask(function()
          writeRenderStatus(outputDir, "started", {
              task_id = taskId,
              output_path = outputPath,
          })
          local taskOk, taskErr = LrTasks.pcall(function()
            local catalog = LrApplication.activeCatalog()
            if not catalog then
                failRender(outputDir, taskId, "active_catalog_missing", "Failed to get active catalog", true)
                return
            end

            logInfo("Creating preset from XMP file: " .. xmpPath)
            local preset = importPreset(xmpPath)
            if not preset then
                failRender(outputDir, taskId, "preset_parse_failed", "Failed to parse settings file: " .. xmpPath, false)
                return
            end

            local shouldUpdateAI = tableContainsMaskSettings(preset)
            local photo = nil
            local processSuccess = catalog:withWriteAccessDo("Import and Process Photo", function()
                logInfo("Searching for photo in catalog: " .. photoPath)
                photo = catalog:findPhotoByPath(photoPath)

                if not photo then
                    logInfo("Photo not found in catalog, attempting to import...")
                    local importedPhoto = catalog:addPhoto(photoPath)
                    if importedPhoto then
                        logInfo("Successfully imported photo")
                        photo = importedPhoto
                        touchImportedPhoto(photoPath)
                        logInfo("Imported photo path: " .. photo:getRawMetadata("path"))
                        logInfo("Imported photo name: " .. photo:getFormattedMetadata("fileName"))
                    end
                else
                    touchImportedPhoto(photoPath)
                end

                if photo then
                    logInfo("Applying preset: " )
                    photo:applyDevelopSettings(preset)

                    if shouldUpdateAI then
                        photo:updateAISettings()
                    else
                        logInfo("Skipping AI settings update for non-mask preset")
                    end
                end

                evictCatalogLru(catalog, {[photoPath] = true})
                writeHealthFile()
            end)

            if not processSuccess then
                failRender(outputDir, taskId, "write_access_failed", "Import/process operation was cancelled or failed", true)
                return
            end

            if not photo then
                failRender(outputDir, taskId, "photo_import_failed", "Could not find or import photo: " .. photoPath, false)
                return
            end
            
            -- Create export settings
            local exportSettings = {
                LR_format = 'JPEG',  -- Export format as JPEG
                LR_jpeg_quality = 0.8,  -- JPEG quality
                LR_export_destinationType = 'specificFolder', 
                LR_export_useSubfolder = false,
                LR_export_destinationPathPrefix = outputDir,  -- Output path
                LR_collisionHandling = 'overwrite',
                LR_renamingTokensOn = true,
                LR_tokenCustomString = photoName,
                LR_tokens = '{{custom_token}}',
            }

            -- -- Execute export
            -- logger:info("Starting export to: " .. outputPath)
            
            -- -- Create export session
            local exportSession = LrExportSession({
                photosToExport = {photo},  -- Single photo
                exportSettings = exportSettings
            })

            writeRenderStatus(outputDir, "export_started", {
                task_id = taskId,
                output_path = outputPath,
            })

            -- Start the export process on a new task
            exportSession:doExportOnNewTask()
            writeRenderStatus(outputDir, "export_requested", {
                task_id = taskId,
                output_path = outputPath,
            })
          end)
          if not taskOk then
              failRender(outputDir, taskId, "plugin_exception", "Photo processing task failed: " .. tostring(taskErr), false)
          end
        end)

        return "processing|" .. outputPath
    end

    if command == "process_batch" and parts[2] and parts[3] then
        local outputDir = parts[2]
        if not outputDir or outputDir == "" then
            return "error|Invalid batch output directory"
        end

        local ok, mkdirErr = pcall(function()
            LrFileUtils.createAllDirectories(outputDir)
        end)
        if not ok then
            logger:error("Failed to create batch output directory: " .. tostring(mkdirErr))
            return "error|Failed to create batch output directory"
        end

        local batchItems = {}
        for i = 3, #parts do
            local item = parts[i]
            local fields = {}
            for field in string.gmatch(item, "[^\t]+") do
                table.insert(fields, field)
            end

            local photoPath = fields[1]
            local xmpPath = fields[2]
            local taskId = fields[3] or tostring(i - 2)

            if not photoPath or photoPath == "" then
                return "error|Invalid photo path in batch"
            end
            if not xmpPath or xmpPath == "" then
                return "error|Invalid XMP path in batch"
            end
            if not LrFileUtils.exists(photoPath) then
                return "error|Photo file does not exist in batch: " .. tostring(taskId)
            end
            if not LrFileUtils.exists(xmpPath) then
                return "error|XMP file does not exist in batch: " .. tostring(taskId)
            end

            local preset = importPreset(xmpPath)
            if not preset then
                return "error|Failed to parse preset in batch: " .. tostring(taskId)
            end

            table.insert(batchItems, {
                photoPath = photoPath,
                xmpPath = xmpPath,
                taskId = taskId,
                preset = preset,
                shouldUpdateAI = tableContainsMaskSettings(preset),
                photo = nil,
            })
        end

        if #batchItems == 0 then
            return "error|Empty batch"
        end

        LrTasks.startAsyncTask(function()
          local taskOk, taskErr = LrTasks.pcall(function()
            local catalog = LrApplication.activeCatalog()
            if not catalog then
                logger:error("Failed to get active catalog")
                return
            end

            local processSuccess = catalog:withWriteAccessDo("Import and Process Batch", function()
                local keepPaths = {}
                for _, item in ipairs(batchItems) do
                    keepPaths[item.photoPath] = true
                    local photo = catalog:findPhotoByPath(item.photoPath)
                    if not photo then
                        photo = catalog:addPhoto(item.photoPath)
                        if photo then
                            touchImportedPhoto(item.photoPath)
                        end
                    else
                        touchImportedPhoto(item.photoPath)
                    end

                    if photo then
                        photo:applyDevelopSettings(item.preset)
                        if item.shouldUpdateAI then
                            photo:updateAISettings()
                        end
                        item.photo = photo
                    else
                        logger:error("Could not find or import batch photo: " .. item.photoPath)
                    end
                end
                evictCatalogLru(catalog, keepPaths)
                writeHealthFile()
            end)

            if not processSuccess then
                logger:error("Batch import/process operation was cancelled or failed")
                return
            end

            local photosToExport = {}
            for _, item in ipairs(batchItems) do
                if item.photo then
                    table.insert(photosToExport, item.photo)
                end
            end

            if #photosToExport == 0 then
                logger:error("Batch had no photos to export")
                return
            end

            local exportSettings = {
                LR_format = 'JPEG',
                LR_jpeg_quality = 0.8,
                LR_export_destinationType = 'specificFolder',
                LR_export_useSubfolder = false,
                LR_export_destinationPathPrefix = outputDir,
            }

            local exportSession = LrExportSession({
                photosToExport = photosToExport,
                exportSettings = exportSettings
            })

            exportSession:doExportOnNewTask()
          end)
          if not taskOk then
              logger:error("Batch processing task failed: " .. tostring(taskErr))
          end
        end)

        return "processing|" .. outputDir
    end
    
    return "error|Invalid request format"
end

local function startServer()
    logInfo("Starting server")
    
    if SERVER_STATUS.running then
        logInfo("Server already running on port " .. SERVER_STATUS.port)
        return
    end
    
    LrTasks.startAsyncTask(function()
        LrFunctionContext.callWithContext("startServer", function(context)
            local running = true
            local server = nil
            
            -- Add cleanup handler
            context:addCleanupHandler(function()
                logInfo("Cleaning up server resources")
                if server then
                    server:close()
                end
                SERVER_STATUS.running = false
                running = false
            end)
            
            -- Start server in async task
            logInfo("Attempting to start server on port " .. SERVER_STATUS.port)
            
            server = LrSocket.bind({
                functionContext = context,
                plugin = _PLUGIN,
                port = SERVER_STATUS.port,
                mode = "receive",
                hostname = "127.0.0.1",
                
                onConnecting = function(socket, port)
                    logInfo("Server connecting on port " .. port)
                end,
                
                onConnected = function(socket, port)
                    logInfo("Server connected on port " .. port)
                    SERVER_STATUS.running = true
                    SERVER_STATUS.error = nil
                end,
                
                onMessage = function(socket, message)
                    logInfo("Received message: " .. tostring(message))
                    -- LrTasks.pcall returns (ok, result) like standard pcall. The old
                    -- code captured only `ok` (a boolean) into `response`, so the next
                    -- lines threw "attempt to concatenate a boolean value" on EVERY
                    -- message. Destructure both and always send back a string.
                    local ok, response = LrTasks.pcall(handleRequest, message)
                    if not ok then
                        logger:error("handleRequest crashed: " .. tostring(response))
                        response = "error|" .. tostring(response)
                    end
                    if response then
                        logInfo("Sending response: " .. tostring(response))
                        socket:send(tostring(response) .. "\n")
                    end
                end,
                
                onClosed = function(socket)
                    logInfo("Client connection closed")
                    -- Server keeps running, restart listening for new connections
                    logInfo("Client disconnected, restarting listener for new connections")
                    socket:reconnect()
                end,
                
                onError = function(socket, err)
                    SERVER_STATUS.error = err
                    -- Restart listening after timeout
                    if err == "timeout" then
                        logInfo("Server timeout, restarting listener...")
                        socket:reconnect()
                        return
                    end
                    -- Log other errors but don't reconnect, keep server stable
                    logger:error("Non-timeout error occurred: " .. tostring(err))
                end,
            })
            
            if not server then
                local errorMsg = "Failed to bind server to port " .. SERVER_STATUS.port
                logger:error(errorMsg)
                error(errorMsg)
            end
            
            logInfo("Server successfully bound to port " .. SERVER_STATUS.port)
            SERVER_STATUS.running = true
            writeHealthFile()
                    
            while running do
                LrTasks.sleep(1/2) 
            end
        end)
    end)
end

-- Periodically ensure the server is up. Use a loop rather than recursion: the
-- previous self-call was not a tail call, so it grew the Lua stack by one frame
-- every 5 minutes and would eventually overflow on a long-running session.
LrTasks.startAsyncTask(function()
    while true do
        if not SERVER_STATUS.running then
            startServer()
        end
        -- Check interval: 5 minutes
        LrTasks.sleep(300)
    end
end)

-- Export module
return {
    getServerStatus = function()
        return SERVER_STATUS
    end,
    startServer = startServer
}
