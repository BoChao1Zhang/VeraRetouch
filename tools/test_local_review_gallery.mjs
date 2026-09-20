// Browser QA through the local headless browser's debugging interface.
import {writeFile} from 'node:fs/promises';
const root='/home/bc/VeraRetouch/outputs/local_review500_20260920';
const page=await (await fetch('http://127.0.0.1:9325/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(page.webSocketDebuggerUrl), pending=new Map();let serial=0;
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
ws.onmessage=e=>{const r=JSON.parse(e.data);if(pending.has(r.id)){const [ok,fail]=pending.get(r.id);pending.delete(r.id);r.error?fail(r.error):ok(r.result)}};
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,[resolve,reject]);ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1200,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:`file://${root}/index.html`});
await evaluate(`new Promise(resolve=>{const t=setInterval(()=>{if(document.querySelectorAll('.card').length){clearInterval(t);resolve(true)}},100)})`);
const initial=await evaluate(`({total:records.length,cards:document.querySelectorAll('.card').length})`);
if(initial.cards!==Math.min(24,initial.total))throw Error('Wrong pagination');
await evaluate(`document.querySelector('[data-select]').click()`);
if(!await evaluate(`chosen.size===1&&document.querySelector('.card').classList.contains('picked')`))throw Error('Selection failed');
await evaluate(`document.querySelector('[data-open]').click()`);
if(!await evaluate(`$('modal').open&&$('detailSelected').checked&&$('detail').querySelectorAll('img').length===17`))throw Error('Detail failed');
await evaluate(`$('nextDetail').click()`);
if(!await evaluate(`active===2&&!$('detailSelected').checked`))throw Error('Detail navigation failed');
await evaluate(`$('close').click();$('view').value='crop';$('view').dispatchEvent(new Event('change'))`);
if(!await evaluate(`document.querySelector('.pair img').src.endsWith('crop_before.png')`))throw Error('Crop switch failed');
await evaluate(`$('onlySelected').click()`);
if(!await evaluate(`document.querySelectorAll('.card').length===1`))throw Error('Selected filter failed');
await evaluate(`window.savedBlob=null;URL.createObjectURL=b=>{window.savedBlob=b;return 'blob:test'};HTMLAnchorElement.prototype.click=function(){};$('export').click()`);
const exported=JSON.parse(await evaluate(`savedBlob.text()`));
if(exported.selected.length!==1||exported.selected[0].number!=='001')throw Error('Export failed');
await evaluate(`$('onlySelected').click();$('search').value=records[3].source_id;$('search').dispatchEvent(new Event('input'))`);
if(!await evaluate(`document.querySelectorAll('.card').length===1`))throw Error('Search failed');
await evaluate(`$('search').value='';$('search').dispatchEvent(new Event('input'));$('view').value='local';$('view').dispatchEvent(new Event('change'));chosen.clear();save();render()`);
await evaluate(`Promise.all([...document.querySelectorAll('.card img')].slice(0,12).map(i=>i.decode()))`);
const screenshot=await call('Page.captureScreenshot',{format:'png'});
await writeFile(`${root}/browser_qa.png`,Buffer.from(screenshot.data,'base64'));
console.log(JSON.stringify({samples:initial.total,checks:['pagination','selection','detail','navigation','crop view','selected filter','export','search'],screenshot:`${root}/browser_qa.png`}));
ws.close();
