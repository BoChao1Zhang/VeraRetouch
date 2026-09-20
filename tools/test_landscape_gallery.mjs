import {writeFile} from 'node:fs/promises';
const root='/home/bc/VeraRetouch/outputs/landscape_review500_20260920';
const tab=await(await fetch('http://127.0.0.1:9326/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(tab.webSocketDebuggerUrl),pending=new Map();let serial=0;
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
ws.onmessage=e=>{const r=JSON.parse(e.data);if(pending.has(r.id)){const [ok,fail]=pending.get(r.id);pending.delete(r.id);r.error?fail(r.error):ok(r.result)}};
const call=(method,params={})=>new Promise((ok,fail)=>{const id=++serial;pending.set(id,[ok,fail]);ws.send(JSON.stringify({id,method,params}))});
async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:`file://${root}/index.html`});
await evaluate(`new Promise(resolve=>{const t=setInterval(()=>{if(document.querySelectorAll('.card').length){clearInterval(t);resolve(true)}},100)})`);
await evaluate(`$('active').checked=false;$('active').dispatchEvent(new Event('change'))`);
for(const s of [2,3,4,5]){
  await evaluate(`$('stage').value='${s}';$('stage').dispatchEvent(new Event('change'))`);
  if(!await evaluate(`document.querySelector('.pair img').src.endsWith('z${s-1}.png')`))throw Error('Wrong before state');
}
await evaluate(`$('stage').value='2';$('stage').dispatchEvent(new Event('change'));document.querySelector('[data-pick]').click();$('stage').value='3';$('stage').dispatchEvent(new Event('change'))`);
if(!await evaluate(`selectedStages[records[0].id]===2`))throw Error('Selected stage was overwritten');
await evaluate(`document.querySelector('[data-open]').click()`);
if(!await evaluate(`$('modal').open&&$('detail').querySelectorAll('img').length===35`))throw Error('Incomplete detail');
await evaluate(`$('detail').querySelectorAll('h2 button')[1].click()`);
if(!await evaluate(`selectedStages[current]===3`))throw Error('Explicit stage selection failed');
await evaluate(`$('close').click();window.savedBlob=null;URL.createObjectURL=b=>{savedBlob=b;return 'blob:test'};HTMLAnchorElement.prototype.click=function(){};$('export').click()`);
const exported=JSON.parse(await evaluate('savedBlob.text()'));
if(exported.selected.length!==1||exported.selected[0].stage!==3)throw Error('Bad export');
await evaluate(`$('view').value='crop';$('view').dispatchEvent(new Event('change'))`);
if(!await evaluate(`document.querySelector('.pair img').src.endsWith('3_before_crop.png')`))throw Error('Bad crop view');
await evaluate(`$('search').value=records[0].id;$('search').dispatchEvent(new Event('input'))`);
if(!await evaluate(`document.querySelectorAll('.card').length===1`))throw Error('Search failed');
await evaluate(`$('search').value='';$('view').value='full';$('stage').value='focus';chosen.clear();selectedStages={};page=0;render()`);
await evaluate(`Promise.all([...document.querySelectorAll('.card img')].slice(0,8).map(i=>i.decode()))`);
const result=await call('Page.captureScreenshot',{format:'png'});await writeFile(root+'/browser_qa.png',Buffer.from(result.data,'base64'));
console.log(JSON.stringify({samples:await evaluate('records.length'),checks:['four stage routing','stage-specific selection','35 detail panels','export','crop view','search']}));ws.close();
