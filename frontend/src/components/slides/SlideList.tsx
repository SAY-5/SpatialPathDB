import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useSlides, useDeleteSlide } from '../../hooks/useSlides';
import { Microscope, Trash2, ArrowRight, ChevronLeft, ChevronRight } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
export default function SlideList() {
  const [page,setPage]=useState(0);
  const {data,isLoading,error}=useSlides(page,12);
  const del=useDeleteSlide();
  if(isLoading)return(<div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'300px',gap:'12px'}}><div className="spinner" style={{width:'18px',height:'18px'}}/><span style={{fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--muted)',letterSpacing:'.1em'}}>LOADING</span></div>);
  if(error)return(<div className="panel" style={{textAlign:'center',padding:'48px'}}><span style={{fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--red)'}}>ERROR: Failed to load slides</span></div>);
  const slides=data?.content||[],total=data?.totalElements||0,totalPages=data?.totalPages||0;
  return(
    <div style={{display:'flex',flexDirection:'column',gap:'20px'}}>
      <div style={{display:'flex',alignItems:'flex-end',justifyContent:'space-between'}}>
        <div><h1 className="page-title">Slides</h1><p className="page-sub">{total.toLocaleString()} specimens in spatial database</p></div>
        {totalPages>1&&<span style={{fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--muted)',paddingBottom:'4px'}}>PAGE {page+1} / {totalPages}</span>}
      </div>
      {slides.length===0?(
        <div className="panel" style={{textAlign:'center',padding:'80px 40px'}}>
          <div style={{width:'44px',height:'44px',border:'1px solid var(--line-2)',display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 16px'}}><Microscope size={20} color="var(--muted)"/></div>
          <div style={{fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--muted)',marginBottom:'8px'}}>NO SLIDES FOUND</div>
          <code style={{fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--cyan)',background:'var(--ink-3)',border:'1px solid var(--line)',padding:'8px 12px',display:'inline-block'}}>python3 scripts/generate_synthetic_data.py --slides 3</code>
        </div>
      ):(
        <>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(290px, 1fr))',gap:'1px',background:'var(--line)'}}>
            {slides.map((slide,i)=>(
              <div key={slide.id} className="slide-card">
                <div style={{padding:'9px 14px',background:'var(--ink-3)',borderBottom:'1px solid var(--line)',display:'flex',alignItems:'center',justifyContent:'space-between'}}>
                  <span style={{fontFamily:'Space Mono, monospace',fontSize:'9px',color:'var(--cyan)',letterSpacing:'.08em'}}>SLIDE_{String(i+1+page*12).padStart(3,'0')}</span>
                  <button onClick={e=>{e.preventDefault();del.mutate(slide.id);}} style={{background:'none',border:'none',cursor:'pointer',padding:'2px',color:'var(--muted)',display:'flex',transition:'color .12s'}} onMouseEnter={e=>(e.currentTarget.style.color='var(--red)')} onMouseLeave={e=>(e.currentTarget.style.color='var(--muted)')}><Trash2 size={11}/></button>
                </div>
                <div style={{padding:'14px',flex:1}}>
                  <div style={{fontFamily:'DM Sans, sans-serif',fontSize:'14px',fontWeight:500,color:'var(--bright)',marginBottom:'12px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{slide.slideName}</div>
                  <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'8px 12px'}}>
                    {([['Organ',slide.organ||'—'],['Stain',slide.stainType||'—'],['Width',`${(slide.widthPixels/1000).toFixed(0)}K px`],['Height',`${(slide.heightPixels/1000).toFixed(0)}K px`],['Resolution',slide.micronsPerPixel?`${slide.micronsPerPixel.toFixed(3)} µm`:'—'],['Added',formatDistanceToNow(new Date(slide.uploadedAt),{addSuffix:true})]] as [string,string][]).map(([label,value])=>(
                      <div key={label}>
                        <div style={{fontFamily:'Space Mono, monospace',fontSize:'9px',color:'var(--muted)',textTransform:'uppercase',letterSpacing:'.08em',marginBottom:'2px'}}>{label}</div>
                        <div style={{fontFamily:'Space Mono, monospace',fontSize:'11px',color:'var(--text)'}}>{value}</div>
                      </div>
                    ))}
                  </div>
                  {slide.objectCount!==undefined&&<div style={{marginTop:'12px',paddingTop:'12px',borderTop:'1px solid var(--line)',display:'flex',alignItems:'center',justifyContent:'space-between'}}><span style={{fontFamily:'Space Mono, monospace',fontSize:'9px',color:'var(--muted)',textTransform:'uppercase',letterSpacing:'.08em'}}>Spatial Objects</span><span style={{fontFamily:'Space Mono, monospace',fontSize:'14px',color:'var(--cyan)',fontWeight:700}}>{slide.objectCount.toLocaleString()}</span></div>}
                </div>
                <Link to={`/slides/${slide.id}`} style={{display:'flex',alignItems:'center',justifyContent:'space-between',padding:'10px 14px',background:'var(--ink-3)',borderTop:'1px solid var(--line)',textDecoration:'none',fontFamily:'Space Mono, monospace',fontSize:'10px',fontWeight:700,letterSpacing:'.08em',textTransform:'uppercase',color:'var(--dim)',transition:'all .12s'}} onMouseEnter={e=>{(e.currentTarget as HTMLElement).style.color='var(--cyan)';(e.currentTarget as HTMLElement).style.background='var(--cyan-dim)';}} onMouseLeave={e=>{(e.currentTarget as HTMLElement).style.color='var(--dim)';(e.currentTarget as HTMLElement).style.background='var(--ink-3)';}}>
                  Open Viewer <ArrowRight size={12}/>
                </Link>
              </div>
            ))}
          </div>
          {totalPages>1&&<div style={{display:'flex',alignItems:'center',justifyContent:'center',gap:'4px'}}><button onClick={()=>setPage(p=>Math.max(0,p-1))} disabled={page===0} className="btn" style={{padding:'6px 10px'}}><ChevronLeft size={14}/></button><span style={{fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--muted)',padding:'0 16px'}}>{page+1} / {totalPages}</span><button onClick={()=>setPage(p=>Math.min(totalPages-1,p+1))} disabled={page>=totalPages-1} className="btn" style={{padding:'6px 10px'}}><ChevronRight size={14}/></button></div>}
        </>
      )}
    </div>
  );
}
