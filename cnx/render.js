const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();
for(const [q,h,n] of [['',1350,(process.argv[3]||'cnx')+'-feed'],['?story',1920,(process.argv[3]||'cnx')+'-story']]){const p=await b.newPage({viewport:{width:1080,height:h}});
await p.goto('file://'+__dirname+'/'+(process.argv[2]||'arte.html')+q);await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(300);
await p.screenshot({path:__dirname+'/'+n+'.png'});}await b.close()})()
