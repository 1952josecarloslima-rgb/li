const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();
for(const [s,n] of [[1,'cnx-telao-1080p'],[2,'cnx-telao-4k']]){const p=await b.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:s});
await p.goto('file://'+__dirname+'/telao.html');await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(400);
await p.screenshot({path:__dirname+'/'+n+'.png'});}await b.close()})()
