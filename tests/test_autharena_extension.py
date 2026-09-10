"""Exercise the companion's actual JavaScript with a mocked Chrome/browser bridge."""
import json
from pathlib import Path
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
