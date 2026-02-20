import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useSlide, useSlideStatistics, useSlideBounds } from '../../hooks/useSlides';
import SpatialViewer from '../spatial/SpatialViewer';
import QueryPanel from '../spatial/QueryPanel';
import StatisticsPanel from '../analytics/StatisticsPanel';
import JobSubmitForm from '../jobs/JobSubmitForm';
import { ArrowLeft, Layers, BarChart2, Zap } from 'lucide-react';
type Tab='viewer'|'statistics'|'jobs';
export default function SlideDetail() {
  const {slideId}=useParams<{slideId:string}>();
  const [tab,setTab]=useState<Tab>('viewer');
  const {data:slide,isLoading}=useSlide(slideId);
  const {data:stats}=useSlideStatistics(slideId);
  const {data:bounds}=useSlideBounds(slideId);
  if(isLoading)return(<div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'300px',gap:'12px'}}><div className="spinner" style={{width:'18px',height:'18px'}}/><span style={{fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--muted)',letterSpacing:'.1em'}}>LOADING SLIDE</span></div>);
  if(!slide)return(<div className="panel" style={{textAlign:'center',padding:'48px'}}><span style={{fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--red)',display:'block',marginBottom:'12px'}}>ERROR: Slide not found</span><Link to="/slides" className="btn-primary">Back to Slides</Link></div>);
  const tabs=[{id:'viewer' as Tab,label:'Viewer',icon:Layers},{id:'statistics' as Tab,label:'Statistics',icon:BarChart2},{id:'jobs' as Tab,label:'Jobs',icon:Zap}];
  return(
    <div style={{display:'flex',flexDirection:'column',gap:'0'}}>
      <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'16px'}}>
        <Link to="/slides" className="btn" style={{padding:'7px 10px'}}><ArrowLeft size={14}/></Link>
        <div style={{flex:1,minWidth:0}}>
          <h1 className="page-title" style={{overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{slide.slideName}</h1>
          <div style={{display:'flex',gap:'12px',marginTop:'3px',fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--muted)'}}>
            <span>{slide.organ}</span><span style={{color:'var(--line-2)'}}>·</span><span>{slide.stainType}</span><span style={{color:'var(--line-2)'}}>·</span><span>{(slide.widthPixels/1000).toFixed(0)}K × {(slide.heightPixels/1000).toFixed(0)}K px</span>
            {bounds?.totalObjects&&<><span style={{color:'var(--line-2)'}}>·</span><span style={{color:'var(--cyan)'}}>{bounds.totalObjects.toLocaleString()} objects</span></>}
          </div>
        </div>
      </div>
      <div style={{display:'flex',borderBottom:'1px solid var(--line)',background:'var(--ink-2)',marginBottom:'16px'}}>
        {tabs.map(({id,label,icon:Icon})=><button key={id} onClick={()=>setTab(id)} className={`tab${tab===id?' active':''}`}><Icon size={12}/>{label}</button>)}
      </div>
      {tab==='viewer'&&<div style={{display:'flex',gap:'12px'}}><div style={{flex:1,minWidth:0}}><div className="panel-flush" style={{height:'600px',overflow:'hidden'}}><SpatialViewer slideId={slideId!} bounds={bounds} slideWidth={slide.widthPixels} slideHeight={slide.heightPixels}/></div></div><div style={{width:'256px',flexShrink:0}}><QueryPanel slideId={slideId!} bounds={bounds}/></div></div>}
      {tab==='statistics'&&<StatisticsPanel slideId={slideId!} statistics={stats}/>}
      {tab==='jobs'&&<JobSubmitForm slideId={slideId!}/>}
    </div>
  );
}
