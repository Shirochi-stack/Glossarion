"""Exercise the companion's actual JavaScript with a mocked Chrome/browser bridge."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


EXTENSION = Path(__file__).resolve().parents[1] / 'assets' / 'autharena_extension'
NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(not NODE, reason='Node.js is required for extension protocol tests')

HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const events = [], requests = [], updates = [], created = [], aborts = [], preparedConfigs = [];
const local = {}, session = {}, tabs = new Map();
let nextTab = 10, dispatches = 0, prepareCount = 0, loginClicks = 0;
let page = {document:'document-1',phase:null,events:[]};
let onDispatch = () => {};
let fetchHook = null;
const listener = () => ({callbacks:[],addListener(fn){this.callbacks.push(fn);}});
const storage = data => ({
  async get(){return {...data};}, async set(value){Object.assign(data,value);}
});
const chrome = {
  storage:{local:storage(local),session:storage(session)},
  runtime:{onMessage:listener(),onStartup:listener(),onInstalled:listener()},
  alarms:{onAlarm:listener(),async create(){}},
  tabs:{onRemoved:listener(),
    async get(id){if(!tabs.has(id))throw Error('Tab closed');return {...tabs.get(id)};},
    async create(options){const tab={id:nextTab++,...options};tabs.set(tab.id,tab);created.push(tab);return tab;},
    async update(id,options){updates.push({id,...options});Object.assign(tabs.get(id),options);return tabs.get(id);}},
  scripting:{async executeScript({target,func,args=[]}){
    let result;
    if(func.name==='pollArenaPage' && !tabs.has(target.tabId))throw Error('Missing tab execution context');
    switch(func.name){
      case 'prepareArena': prepareCount++; preparedConfigs.push(args[0]); page.phase=args[0].login_only?'done':'ready';
        page.events.push({event:'verified',logged_in:true,tou_accepted:true},
          {event:args[0].login_only?'done':'ready'});break;
      case 'tagArenaJob': page.job=args[0];break;
      case 'pollArenaPage':result={arena:true,loaded:true,document:page.document,phase:page.phase,
        rejections:0,events:page.events.splice(0,64)};break;
      case 'dispatchArenaJob':dispatches++;page.phase='dispatched';onDispatch();result=true;break;
      case 'abortArenaJob':aborts.push(args[0]);page.phase='cancelled';break;
      case 'openArenaLogin':loginClicks++;result=true;break;
      default:throw Error('Unexpected injected function '+func.name);
    }
    return [{result}];
  }}
};
const context=vm.createContext({chrome,URL,AbortController,crypto:require('crypto').webcrypto,
  setTimeout,clearTimeout,console,Date,Promise,
  importScripts(){},prepareArena:function prepareArena(){},
  fetch:async(url,options)=>{
    const body=options.body?JSON.parse(options.body):null;
    requests.push({url,headers:options.headers,body});
    if(fetchHook){const result=await fetchHook(url,options,body);if(result)return result;}
    if(url.endsWith('/extension/events')){events.push(...body.events.map(event=>({job:body.job_id,...event})));return {ok:true,status:200,json:async()=>({})};}
    if(url.endsWith('/pair'))return {ok:true,status:200,json:async()=>({device_id:body.device_id,device_token:'paired-secret',account_id:3})};
    throw Error('Bridge offline');
  }
});
const run=code=>vm.runInContext(code,context);
const tick=()=>new Promise(resolve=>setTimeout(resolve,10));
const until=async condition=>{for(let i=0;i<250;i++){if(condition())return;await tick();}throw Error('Condition timed out');};
__SETUP__
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{
  await tick();
  run("device={device_id:'device-1',device_token:'secret',account_id:0}");
  const result=await (async()=>{ __TEST__ })();
  console.log(JSON.stringify(result));
  process.exit(0);
})().catch(error=>{console.error(error.stack);process.exit(1);});
"""


def run_js(script, setup=''):
    result = subprocess.run(
        [NODE, '-e', HARNESS.replace('__SETUP__', setup).replace('__TEST__', script), str(EXTENSION / 'background.js')],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def run_connect_page(script, setup=''):
    import autharena_bridge

    html = autharena_bridge._connect_page(Path('C:/Glossarion/Arena helper'))
    source = re.search(r'<script>(.*?)</script>', html, re.S).group(1)
    harness = r"""
const vm=require('vm');
const elements=new Map(), timers=[], requests=[], messages=[], stored=new Map();
const responseQueue=[];
let reloaded=0, replaced=0, pairResult={ok:true,account_id:0};
function element(id){if(!elements.has(id))elements.set(id,{textContent:'',disabled:false,open:false,
  listeners:{},addEventListener(name,fn){this.listeners[name]=fn;}});return elements.get(id);}
element('folder').textContent='C:/Glossarion/Arena helper';
const location={origin:'http://127.0.0.1:18874',pathname:'/connect',search:'',
  hash:'#abcdefghijklmnopqrstuvwx',reload(){reloaded++;}};
const listeners={};
const context=vm.createContext({
  location,URL,Promise,console,
  navigator:{userAgent:'Mozilla Chrome/140.0 Edg/140.0',clipboard:{async writeText(value){messages.push({copied:value});}}},
  document:{getElementById:element,createElement(){return {setAttribute(){},textContent:''};},body:{appendChild(){}}},
  sessionStorage:{getItem:key=>stored.get(key)||null,setItem:(key,value)=>stored.set(key,value),removeItem:key=>stored.delete(key)},
  history:{replaceState(){replaced++;location.hash='';}},
  setTimeout(fn,delay){timers.push({fn,delay});},
  addEventListener(name,fn){listeners[name]=fn;},
  chrome:{runtime:{async sendMessage(message){requests.push({pair:message});return pairResult;}}},
  fetch:async(path,options)=>{
    requests.push({path,options,body:JSON.parse(options.body)});
    const response=responseQueue.shift()||{status:'ready',message:'Ready'};
    return {ok:!response.error,json:async()=>response};
  }
});
context.window=context;
const run=source=>vm.runInContext(source,context);
context.deliverMessage=event=>listeners.message?.(event);
context.postMessage=(data,origin)=>{
  messages.push(data);context.messageData=data;context.messageOrigin=origin;
  run('deliverMessage({source:window,origin:messageOrigin,data:messageData})');
};
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
(async()=>{
  __SETUP__
  run(__SOURCE__);
  const result=await(async()=>{__TEST__})();
  console.log(JSON.stringify(result));
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
"""
    program = harness.replace('__SOURCE__', json.dumps(source)).replace('__SETUP__', setup).replace('__TEST__', script)
    result = subprocess.run([NODE, '-e', program], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_connect_page_requires_click_then_posts_only_nonce_and_browser_hint():
    result = run_connect_page(r"""
      await timers.shift().fn();
      const before=requests.map(item=>item.path);
      responseQueue.push({status:'running',message:'Installing',attempt_id:'attempt-1'});
      await element('install').listeners.click();
      await element('install').listeners.click();
      return {before,requests,disabled:element('install').disabled,reloaded};
    """)
    assert result['before'] == ['/setup/status']
    installs = [request for request in result['requests'] if request['path'] == '/setup/install']
    assert len(installs) == 1
    assert installs[0]['body'] == {'nonce': 'abcdefghijklmnopqrstuvwx', 'browser_hint': 'edge'}
    assert installs[0]['options']['credentials'] == 'omit'
    assert result['disabled'] and result['reloaded'] == 0


def test_connect_page_reconnects_after_setup_without_restarting_installer():
    result = run_connect_page(r"""
      responseQueue.push({status:'awaiting_connection',message:'Waiting for helper',attempt_id:'attempt-1'});
      await timers.shift().fn();
      const message=element('status').textContent;
      await timers.shift().fn();
      return {requests,reloaded,stored:[...stored.values()],message};
    """)
    assert [item['path'] for item in result['requests']] == ['/setup/status']
    assert result['reloaded'] == 1
    assert json.loads(result['stored'][0]) == {'attempt': 'attempt-1', 'count': 1}
    assert result['message'] == 'Waiting for helper'


def test_connect_page_caps_reload_attempts_and_shows_manual_help():
    result = run_connect_page(r"""
      responseQueue.push({status:'awaiting_connection',message:'Waiting for helper',attempt_id:'attempt-1'});
      await timers.shift().fn();
      return {reloaded,timers:timers.length,manual:element('manual').open,message:element('status').textContent};
    """, setup=r"""
      stored.set('autharena-setup:abcdefghijklmnopqrstuvwx',JSON.stringify({attempt:'attempt-1',count:15}));
    """)
    assert result['reloaded'] == result['timers'] == 0
    assert result['manual'] and 'has not connected yet' in result['message']


def test_connect_page_expired_nonce_stops_reconnect_and_install():
    result = run_connect_page(r"""
      responseQueue.push({error:'Login expired'});
      await timers.shift().fn();
      await element('install').listeners.click();
      return {requests,reloaded,timers:timers.length,message:element('status').textContent};
    """)
    assert [item['path'] for item in result['requests']] == ['/setup/status']
    assert result['reloaded'] == result['timers'] == 0
    assert result['message'] == 'Login expired'


def test_connect_page_renders_installer_errors_as_text_and_never_success():
    result = run_connect_page(r"""
      responseQueue.push({status:'manual_required',message:'<img src=x onerror=alert(1)>'});
      await timers.shift().fn();
      return {manual:element('manual').open,message:element('status').textContent,reloaded};
    """)
    assert result['message'] == '<img src=x onerror=alert(1)>'
    assert result['manual'] is True and result['reloaded'] == 0


def test_connect_page_confirms_actual_pairing_and_stops_pending_reloads():
    result = run_connect_page(r"""
      responseQueue.push({status:'awaiting_connection',message:'Waiting',attempt_id:'attempt-1'});
      await timers.shift().fn();
      context.postMessage({type:'arena-pair-result',ok:true,account_id:0},location.origin);
      await timers.shift().fn();
      await element('install').listeners.click();
      return {reloaded,label:element('install').textContent,message:element('status').textContent,
        disabled:element('install').disabled,stored:[...stored],requests};
    """)
    assert result['reloaded'] == 0 and result['stored'] == []
    assert result['label'] == 'Helper connected' and result['disabled']
    assert 'Opening Arena sign-in' in result['message']
    assert [item['path'] for item in result['requests']] == ['/setup/status']


def test_connect_page_rejects_pair_success_messages_from_other_origins():
    result = run_connect_page(r"""
      context.postMessage({type:'arena-pair-result',ok:true},'https://evil.example');
      return {label:element('install').textContent,disabled:element('install').disabled,message:element('status').textContent};
    """)
    assert result['label'] != 'Helper connected' and not result['disabled']
    assert 'Opening Arena sign-in' not in result['message']


@pytest.mark.parametrize('success', [True, False])
def test_content_script_keeps_nonce_until_pairing_succeeds(success):
    content_script = (EXTENSION / 'connect.js').read_text(encoding='utf-8')
    result = run_connect_page(
        'pairResult=' + json.dumps({'ok': success, 'account_id': 0}) + ';run(' + json.dumps(content_script) + r""");
      await tick();
      return {replaced,hash:location.hash,requests,messages};
    """)
    assert result['replaced'] == (1 if success else 0)
    assert bool(result['hash']) is not success
    pair = next(item for item in result['requests'] if 'pair' in item)
    assert pair['pair']['nonce'] == 'abcdefghijklmnopqrstuvwx'
    assert all('device_token' not in str(message) for message in result['messages'])


def test_companion_permissions_are_limited_to_arena_and_local_bridge():
    manifest = json.loads((EXTENSION / 'manifest.json').read_text(encoding='utf-8'))
    assert set(manifest['permissions']) == {'storage', 'tabs', 'scripting', 'alarms'}
    assert set(manifest['host_permissions']) == {'https://arena.ai/*', 'http://127.0.0.1:18874/*'}
    assert 'cookies' not in manifest['permissions']
    assert 'debugger' not in manifest['permissions']


def test_pairing_checks_sender_and_keeps_device_token_out_of_page_response():
    result = run_js(r"""
      const accepted=run("validConnectSender({url:'http://127.0.0.1:18874/connect#nonce',tab:{id:1}})");
      const rejected=['https://arena.ai/connect','http://localhost:18874/connect','http://127.0.0.1:18875/connect','http://127.0.0.1:18874/connect/evil'];
      const invalid=rejected.map(url=>context.validConnectSender({url,tab:{id:1}}));
      const response=await context.pair('abcdefghijklmnopqrstuvwx');
      await tick();
      return {accepted,invalid,response,stored:local.autharenaDevice,
        pairRequest:requests.find(r=>r.url.endsWith('/pair'))};
    """)
    assert result['accepted']
    assert result['invalid'] == [False] * 4
    assert result['response'] == {'ok': True, 'account_id': 3}
    assert result['stored']['device_token'] == 'paired-secret'
    assert result['pairRequest']['body']['device_id'] == result['stored']['device_id']
    assert 'Authorization' not in result['pairRequest']['headers']


def test_early_pairing_and_startup_poll_wait_for_storage_recovery():
    result = run_js(r"""
      let pairResponse;
      const accepted=chrome.runtime.onMessage.callbacks[0](
        {type:'autharena-pair',nonce:'abcdefghijklmnopqrstuvwx'},
        {url:'http://127.0.0.1:18874/connect',tab:{id:1}},result=>{pairResponse=result;});
      chrome.runtime.onStartup.callbacks[0]();
      chrome.alarms.onAlarm.callbacks[0]({name:'autharena-reconnect'});
      await tick();
      const beforeRecovery={requests:requests.length,responded:pairResponse!==undefined};
      releaseInitialization();
      await until(()=>pairResponse!==undefined);
      await tick();
      return {accepted,beforeRecovery,pairResponse,device:run('device'),stored:local.autharenaDevice,
        seen:run('seenJobs'),aborts,polls,recovered:local.autharenaActiveJob};
    """, setup=r"""
      local.autharenaDevice={device_id:'retained-id'};
      local.autharenaSeenJobs=['already-consumed'];
      local.autharenaActiveJob={id:'interrupted',tabId:8,dispatched:true};
      const startupSnapshot=structuredClone(local), originalGet=chrome.storage.local.get;
      let releaseInitialization;
      chrome.storage.local.get=async keys=>Array.isArray(keys)
        ? new Promise(resolve=>{releaseInitialization=()=>resolve(startupSnapshot);})
        : originalGet(keys);
      const polls=[];
      fetchHook=async(url,options)=>{
        if(url.endsWith('/extension/poll'))polls.push({authorization:options.headers.Authorization,
          recovered:local.autharenaActiveJob===null,aborted:aborts.includes('interrupted'),
          seen:[...local.autharenaSeenJobs]});
        return null;
      };
    """)
    assert result['accepted'] is True
    assert result['beforeRecovery'] == {'requests': 0, 'responded': False}
    assert result['pairResponse'] == {'ok': True, 'account_id': 3}
    assert result['device'] == result['stored'] == {
        'device_id': 'retained-id', 'device_token': 'paired-secret', 'account_id': 3,
    }
    assert result['seen'] == ['already-consumed', 'interrupted']
    assert result['aborts'] == ['interrupted']
    assert result['recovered'] is None
    assert result['polls'] == [{
        'authorization': 'Bearer paired-secret', 'recovered': True, 'aborted': True,
        'seen': ['already-consumed', 'interrupted'],
    }]


def test_login_uses_homepage_and_does_not_overwrite_user_chat():
    result = run_js(r"""
      tabs.set(7,{id:7,url:'https://arena.ai/c/user-chat'});
      run("ownedTab={id:7,url:'https://arena.ai/'}");
      await context.acceptCommand({type:'run',job_id:'login-1',config:{login:true,timeout:2}});
      await until(()=>events.some(e=>e.event==='done'));
      return {created,updates,loginClicks,dispatches,old:tabs.get(7),events,preparedConfigs};
    """)
    assert result['created'][0]['url'] == 'https://arena.ai/'
    assert result['old']['url'] == 'https://arena.ai/c/user-chat'
    assert all(update['id'] != 7 for update in result['updates'])
    assert result['loginClicks'] >= 1
    assert result['dispatches'] == 0
    assert result['preparedConfigs'][0]['model'] == ''
    assert [event['event'] for event in result['events']] == ['verified', 'done']


def test_stream_waits_for_dispatch_and_forwards_chunks_before_completion():
    result = run_js(r"""
      await context.acceptCommand({type:'run',job_id:'stream-1',config:{model:'model/name',payload:{},timeout:3}});
      await until(()=>events.some(e=>e.event==='ready'));
      const before=dispatches;
      onDispatch=()=>page.events.push({event:'chunk',data:'first'});
      await context.acceptCommand({type:'dispatch',job_id:'stream-1'});
      await context.acceptCommand({type:'dispatch',job_id:'stream-1'});
      await until(()=>events.some(e=>e.event==='chunk'));
      const beforeDone=events.filter(e=>e.event==='chunk').map(e=>e.data);
      const doneWasSent=events.some(e=>e.event==='done');
      page.events.push({event:'chunk',data:'second'},{event:'done'});page.phase='done';
      await until(()=>events.some(e=>e.event==='done'));
      await context.acceptCommand({type:'run',job_id:'stream-1',config:{model:'model/name'}});
      return {before,beforeDone,doneWasSent,events,dispatches,prepareCount,created,requests};
    """)
    assert result['before'] == 0
    assert result['beforeDone'] == ['first']
    assert not result['doneWasSent']
    assert result['dispatches'] == result['prepareCount'] == 1
    assert [e['data'] for e in result['events'] if e['event'] == 'chunk'] == ['first', 'second']
    assert result['created'][0]['url'].endswith('model_a=model%2Fname')
    assert all(r['headers']['Authorization'] == 'Bearer secret' for r in result['requests'])


def test_navigation_after_dispatch_is_terminal_and_never_replays():
    result = run_js(r"""
      await context.acceptCommand({type:'run',job_id:'navigation',config:{model:'model',payload:{},timeout:2}});
      await until(()=>events.some(e=>e.event==='ready'));
      onDispatch=()=>{page.document='new-document';};
      await context.acceptCommand({type:'dispatch',job_id:'navigation'});
      await until(()=>events.some(e=>e.event==='error'));
      await context.acceptCommand({type:'run',job_id:'navigation',config:{model:'model'}});
      return {events,dispatches,prepareCount,aborts};
    """)
    error = next(e for e in result['events'] if e['event'] == 'error')
    assert error['request_dispatched'] is True
    assert result['dispatches'] == result['prepareCount'] == 1
    assert result['aborts'] == ['navigation']


def test_lost_bridge_cancels_active_generation_and_preserves_recovery_error():
    result = run_js(r"""
      await context.acceptCommand({type:'run',job_id:'offline',config:{model:'model',payload:{},timeout:3}});
      await until(()=>events.some(e=>e.event==='ready'));
      await context.acceptCommand({type:'dispatch',job_id:'offline'});
      await context.startPolling();
      await context.acceptCommand({type:'run',job_id:'offline',config:{model:'model'}});
      return {aborts,dispatches,pending:local.autharenaPendingErrors,active:run('activeJob')};
    """)
    assert result['active'] is None
    assert result['dispatches'] == 1
    assert result['aborts'] == ['offline']
    assert result['pending'][0]['events'][0]['request_dispatched'] is True


def test_worker_restart_cancels_persisted_job_without_resubmission():
    result = run_js(r"""
      local.autharenaActiveJob={id:'interrupted',tabId:8,dispatched:true};
      local.autharenaSeenJobs=['interrupted'];
      await context.initialize();
      await context.acceptCommand({type:'run',job_id:'interrupted',config:{model:'model'}});
      return {aborts,dispatches,created,pending:local.autharenaPendingErrors};
    """)
    assert result['aborts'] == ['interrupted']
    assert result['dispatches'] == 0
    assert result['created'] == []
    assert result['pending'][0]['events'][0]['request_dispatched'] is True


def test_cancel_before_dispatch_remains_unsubmitted():
    result = run_js(r"""
      await context.acceptCommand({type:'run',job_id:'cancel',config:{model:'model',payload:{},timeout:2}});
      await until(()=>events.some(e=>e.event==='ready'));
      await context.acceptCommand({type:'cancel',job_id:'cancel'});
      return {events,dispatches,aborts};
    """)
    error = next(e for e in result['events'] if e['event'] == 'error')
    assert error['request_dispatched'] is False
    assert 'cancelled' in error['message']
    assert result['dispatches'] == 0


def test_login_click_preserves_google_email_choice_and_ignores_hidden_controls():
    result = run_js(r"""
      const clicked=[];
      const button=(text,visible=true)=>({textContent:text,disabled:false,getAttribute:()=>null,
        getClientRects:()=>visible?[{}]:[],closest:()=>null,click:()=>clicked.push(text)});
      context.location={origin:'https://arena.ai'};
      context.document={querySelectorAll:selector=>selector.includes('dialog')?[]:[button('Log In',false),button('Continue with Google'),button('Continue with email'),button('Log In')]};
      context.getComputedStyle=()=>({visibility:'visible',display:'block'});
      const found=context.openArenaLogin();
      return {found,clicked};
    """)
    assert result == {'found': True, 'clicked': ['Log In']}


@pytest.mark.parametrize('modal_open', [True, False])
def test_login_opener_never_clicks_existing_credential_form(modal_open):
    result = run_js(r"""
      const clicked=[];
      const submit={textContent:'Log In',getAttribute:()=>null,closest:()=>({tagName:'FORM'}),
        getClientRects:()=>[{}],click:()=>clicked.push('submit')};
      const dialog={getClientRects:()=>[{}]};
      context.location={origin:'https://arena.ai'};
      context.document={querySelectorAll:selector=>selector.includes('dialog')?(__MODAL__?[dialog]:[]):[submit]};
      context.getComputedStyle=()=>({visibility:'visible',display:'block'});
      const found=context.openArenaLogin();
      return {found,clicked};
    """.replace('__MODAL__', json.dumps(modal_open)))
    assert result['clicked'] == []
    assert result['found'] is modal_open


def test_confirmed_closed_tab_before_dispatch_reports_cancellation():
    result = run_js(r"""
      await context.acceptCommand({type:'run',job_id:'closed',config:{model:'model',payload:{},timeout:2}});
      await until(()=>created.length===1);
      tabs.delete(created[0].id); // onRemoved can be delayed; executeScript must confirm closure itself.
      await until(()=>events.some(e=>e.event==='error'));
      return {error:events.find(e=>e.event==='error'),dispatches,created};
    """)
    assert result['error']['error_type'] == 'cancelled'
    assert 'tab was closed' in result['error']['message']
    assert result['error']['request_dispatched'] is False
    assert result['dispatches'] == 0
    assert len(result['created']) == 1


@pytest.mark.parametrize('dispatched', [False, True])
def test_companion_deadline_preserves_timeout_type_and_dispatch_state(dispatched):
    result = run_js(r"""
      await context.acceptCommand({type:'run',job_id:'timeout',config:{model:'model',payload:{},timeout:.25}});
      await until(()=>events.some(e=>e.event==='ready'));
      if(__DISPATCHED__)await context.acceptCommand({type:'dispatch',job_id:'timeout'});
      await until(()=>events.some(e=>e.event==='error'));
      return events.find(e=>e.event==='error');
    """.replace('__DISPATCHED__', json.dumps(dispatched)))
    assert result['error_type'] == 'timeout'
    assert result['request_dispatched'] is dispatched


@pytest.mark.parametrize('status', [404, 410])
def test_expired_recovery_error_does_not_block_reconnect_or_erase_consumed_id(status):
    result = run_js(r"""
      run("pendingErrors=[{job_id:'expired',events:[{event:'error',request_dispatched:true}]}];seenJobs=['expired']");
      fetchHook=async(url)=>url.endsWith('/extension/events')?{ok:false,status:__STATUS__}:null;
      await context.startPolling();
      await context.acceptCommand({type:'run',job_id:'expired',config:{model:'model'}});
      return {pending:run('pendingErrors'),seen:run('seenJobs'),created,
        polled:requests.some(r=>r.url.endsWith('/extension/poll'))};
    """.replace('__STATUS__', str(status)))
    assert result == {'pending': [], 'seen': ['expired'], 'created': [], 'polled': True}


def test_pair_conflict_explains_browser_profile_requirement():
    result = run_js(r"""
      fetchHook=async()=>({ok:false,status:409});
      const response=await new Promise(resolve=>chrome.runtime.onMessage.callbacks[0](
        {type:'autharena-pair',nonce:'abcdefghijklmnopqrstuvwx'},
        {url:'http://127.0.0.1:18874/connect',tab:{id:1}},resolve));
      return response;
    """)
    assert not result['ok']
    assert 'different browser profile' in result['error']


@pytest.mark.parametrize('path,expected_calls', [('/connect', 1), ('/connect/elsewhere', 0)])
def test_connect_script_clears_nonce_and_reports_pairing_without_credentials(path, expected_calls):
    script = r"""
      const sent=[],published=[],history=[];
      context.location={origin:'http://127.0.0.1:18874',pathname:__PATH__,hash:'#abcdefghijklmnopqrstuvwx',search:''};
      context.history={replaceState:(state,title,url)=>history.push(url)};
      context.window={postMessage:(data,origin)=>published.push({data,origin})};
      context.document={createElement:()=>({setAttribute(){}}),body:{appendChild(){}}};
      chrome.runtime.sendMessage=async message=>{sent.push(message);return {ok:true,account_id:2,device_token:'must-not-reach-page'};};
      vm.runInContext(fs.readFileSync(__CONNECT_SOURCE__,'utf8'),context);
      await tick();
      return {sent,published,history};
    """.replace('__PATH__', json.dumps(path)).replace('__CONNECT_SOURCE__', json.dumps(str(EXTENSION / 'connect.js')))
    result = run_js(script)
    assert len(result['sent']) == expected_calls
    if expected_calls:
        assert result['history'] == ['/connect']
        assert result['published'] == [{'origin': 'http://127.0.0.1:18874',
                                       'data': {'type': 'arena-pair-result', 'ok': True, 'account_id': 2}}]
    else:
        assert result == {'sent': [], 'published': [], 'history': []}
