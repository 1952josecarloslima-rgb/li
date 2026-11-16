const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();
for(const [q,h,n] of [['',1350,'cnx-feed'],['?story',1920,'cnx-story']]){const p=await b.newPage({viewport:{width:1080,height:h}});
await p.goto('file://'+__dirname+'/arte.html'+q);await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(300);
await p.screenshot({path:__dirname+'/'+n+'.png'});}await b.close()})()
