const {chromium}=require('playwright');const fs=require('fs');(async()=>{
const [,,fps,dur,out,only]=process.argv;const b=await chromium.launch();const p=await b.newPage({viewport:{width:1920,height:1080}});
await p.goto('file://'+__dirname+'/telao-video.html');await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(500);
fs.mkdirSync(out,{recursive:true});const N=Math.round(fps*dur);
for(let i=0;i<N;i++){if(only&&!only.split(',').includes(String(i)))continue;await p.evaluate(t=>render(t),i/fps);await p.screenshot({path:`${out}/f${String(i).padStart(4,'0')}.jpg`,type:'jpeg',quality:95});}
await b.close()})()
