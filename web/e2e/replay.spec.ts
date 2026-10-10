import {test, expect} from '@playwright/test';

test('real local recording supports model comparison and export',async({page})=>{
  test.skip(!process.env.ACTIVITY_DEMO_URL,'Requires prepared local data and serving packages');
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'Watch the evidence unfold.'})).toBeVisible();
  await page.getByRole('button',{name:'Analyze recording',exact:true}).click();
  await expect(page.getByRole('status')).toHaveText('24/24 conditions analyzed',{timeout:60000});
  await expect(page.locator('.prediction .decision')).toHaveCount(6);
  await page.getByRole('button',{name:'100%',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Predictions after 100% of the recording'})).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button',{name:'Export predictions'}).click();
  expect((await download).suggestedFilename()).toContain('-predictions.json');
  await page.getByLabel('Recording',{exact:true}).selectOption('a22_s2_t1');
  await expect(page.locator('.pill')).toHaveText('Held-out action');
  await expect(page.locator('.prediction .decision')).toHaveCount(0);
  await page.getByRole('button',{name:'Play replay'}).click();
  await expect(page.getByRole('button',{name:'Pause replay'})).toBeVisible();
  await page.getByRole('button',{name:'Pause replay'}).click();
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});

test('failed and partial analysis never reveals later predictions',async({page})=>{
  test.skip(!!process.env.ACTIVITY_DEMO_URL,'Uses generated API fixtures');
  const trial={id:'a1_s2_t1',action:1,subject:2,downstream_known_action:true};
  await page.route('**/replay',r=>r.fulfill({json:[trial]}));
  await page.route('**/models',r=>r.fulfill({json:{robust:{family:'robust',sha256:'a'.repeat(64),limitations:[]}}}));
  await page.route('**/replay/a1_s2_t1',r=>r.fulfill({json:{...trial,frame_count:12,fps:15,duration_seconds:.8,imu:Array.from({length:20},(_,i)=>[i,0,1,0,i,2]),alignment:'Normalized trial fraction',sensor_placement:'wrist'}}));
  await page.route('**/frames/*',r=>r.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"/>'}));
  let calls=0;
  await page.route('**/predict/**',r=>{
    calls++;
    if(calls===4)return r.fulfill({status:503,json:{detail:'Inference busy; retry later'}});
    return r.fulfill({json:{trial:trial.id,top_action:1,accepted:true,confidence:.7,threshold:.5,model_sha256:'a'.repeat(64),fraction:.25,labels:[1],probabilities:[.7]}});
  });
  await page.goto('/');await page.getByRole('button',{name:'Analyze recording',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('Inference busy');
  await expect(page.locator('.decision')).toHaveCount(3);
  await page.getByRole('button',{name:'50%',exact:true}).click();
  await expect(page.locator('.decision')).toHaveCount(0);
  await expect(page.locator('.empty')).toHaveCount(3);
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});
