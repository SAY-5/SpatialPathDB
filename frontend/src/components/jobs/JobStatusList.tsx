import { useState } from 'react';
import { useJobs, useJobCounts, useCancelJob } from '../../hooks/useJobs';
import { formatDistanceToNow } from 'date-fns';
import { Zap, CheckCircle, XCircle, Clock, Loader, Ban, ChevronLeft, ChevronRight } from 'lucide-react';
import type { AnalysisJob } from '../../types';
const SC:Record<AnalysisJob['status'],{icon:typeof Zap;bc:string}>={QUEUED:{icon:Clock,bc:'badge badge-queued'},RUNNING:{icon:Loader,bc:'badge badge-running'},COMPLETED:{icon:CheckCircle,bc:'badge badge-completed'},FAILED:{icon:XCircle,bc:'badge badge-failed'},CANCELLED:{icon:Ban,bc:'badge badge-cancelled'}};
const SA:Record<string,string>={QUEUED:'var(--dim)',RUNNING:'var(--amber)',COMPLETED:'var(--green)',FAILED:'var(--red)',CANCELLED:'var(--muted)'};
export default function JobStatusList() {
  const [page,setPage]=useState(0);
  const [sf,setSf]=useState<string|undefined>();
  const {data:jd,isLoading}=useJobs(page,20,sf);
  const {data:counts}=useJobCounts();
  const cancel=useCancelJob();
  const jobs=jd?.content||[],totalPages=jd?.totalPages||0;
  return(
    <div style={{display:'flex',flexDirection:'column',gap:'20px'}}>
      <div><h1 className="page-title">Analysis Jobs</h1><p className="page-sub">{jd?.totalElements||0} total jobs</p></div>
      {counts&&<div style={{display:'flex',gap:'1px',background:'var(--line)'}}>
        {Object.entries(counts).map(([s,c])=>{const u=s.toUpperCase() as AnalysisJob['status'];const a=sf===u;const col=SA[u]||'var(--dim)';return(
          <button key={s} onClick={()=>setSf(a?undefined:u)} style={{flex:1,padding:'12px 16px',background:a?'var(--ink-3)':'var(--ink-2)',border:'none',cursor:'pointer',textAlign:'left',transition:'background .12s',borderBottom:a?`2px solid ${col}`:'2px solid transparent'}}>
            <div style={{fontFamily:'Space Mono, monospace',fontSize:'20px',fontWeight:700,color:a?col:'var(--bright)',lineHeight:1,marginBottom:'4px'}}>{c}</div>
            <div style={{fontFamily:'Space Mono, monospace',fontSize:'9px',fontWeight:700,letterSpacing:'.1em',textTransform:'uppercase',color:a?col:'var(--muted)'}}>{s}</div>
          </button>
        );})}
      </div>}
      {isLoading?<div style={{display:'flex',alignItems:'center',justifyContent:'center',padding:'80px',gap:'12px'}}><div className="spinner" style={{width:'18px',height:'18px'}}/><span style={{fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--muted)'}}>LOADING</span></div>
      :jobs.length===0?<div className="panel" style={{textAlign:'center',padding:'80px'}}><Zap size={28} color="var(--muted)" style={{margin:'0 auto 12px',display:'block',opacity:.3}}/><div style={{fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--muted)'}}>NO JOBS FOUND</div></div>
      :<><div className="panel-flush">
        <div style={{display:'grid',gridTemplateColumns:'110px 1fr 130px 160px 90px 80px',borderBottom:'1px solid var(--line)',background:'var(--ink-3)'}}>
          {['JOB ID','TYPE','STATUS','SUBMITTED','DURATION',''].map((h,i)=><div key={i} style={{padding:'8px 14px',fontFamily:'Space Mono, monospace',fontSize:'9px',fontWeight:700,letterSpacing:'.1em',textTransform:'uppercase',color:'var(--muted)'}}>{h}</div>)}
        </div>
        {jobs.map((job,i)=>{const cfg=SC[job.status];const Icon=cfg.icon;return(
          <div key={job.id} style={{display:'grid',gridTemplateColumns:'110px 1fr 130px 160px 90px 80px',borderBottom:i<jobs.length-1?'1px solid var(--line)':'none',transition:'background .1s'}} onMouseEnter={e=>(e.currentTarget.style.background='var(--ink-3)')} onMouseLeave={e=>(e.currentTarget.style.background='transparent')}>
            <div style={{padding:'11px 14px',fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--dim)'}}>{job.id.slice(0,8)}</div>
            <div style={{padding:'11px 14px',fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--text)'}}>{job.jobType}</div>
            <div style={{padding:'8px 14px',display:'flex',alignItems:'center'}}><span className={cfg.bc}><Icon size={8} style={{animation:job.status==='RUNNING'?'spin 1s linear infinite':'none'}}/>{job.status}</span></div>
            <div style={{padding:'11px 14px',fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--muted)'}}>{formatDistanceToNow(new Date(job.submittedAt),{addSuffix:true})}</div>
            <div style={{padding:'11px 14px',fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--dim)',textAlign:'right'}}>{job.durationMs?`${(job.durationMs/1000).toFixed(1)}s`:'—'}</div>
            <div style={{padding:'8px 14px',display:'flex',alignItems:'center',justifyContent:'flex-end'}}>{(job.status==='QUEUED'||job.status==='RUNNING')&&<button onClick={()=>cancel.mutate(job.id)} className="btn-danger">STOP</button>}</div>
          </div>
        );})}
      </div>
      {totalPages>1&&<div style={{display:'flex',alignItems:'center',justifyContent:'center',gap:'4px'}}><button onClick={()=>setPage(p=>Math.max(0,p-1))} disabled={page===0} className="btn" style={{padding:'6px 10px'}}><ChevronLeft size={14}/></button><span style={{fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--muted)',padding:'0 16px'}}>{page+1} / {totalPages}</span><button onClick={()=>setPage(p=>Math.min(totalPages-1,p+1))} disabled={page>=totalPages-1} className="btn" style={{padding:'6px 10px'}}><ChevronRight size={14}/></button></div>}
      </>}
    </div>
  );
}
