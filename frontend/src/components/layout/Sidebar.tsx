import { NavLink } from 'react-router-dom';
import { Microscope, Zap, BarChart3 } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { healthApi } from '../../api/client';
const nav = [{to:'/slides',icon:Microscope,label:'Slides'},{to:'/jobs',icon:Zap,label:'Jobs'},{to:'/benchmarks',icon:BarChart3,label:'Benchmarks'}];
function Dot({up}:{up:boolean}){return <div style={{width:'5px',height:'5px',borderRadius:'50%',flexShrink:0,background:up?'var(--green)':'var(--red)',boxShadow:up?'0 0 5px var(--green)':'none'}}/>;}
export default function Sidebar() {
  const {data:h} = useQuery({queryKey:['health'],queryFn:()=>healthApi.check(),refetchInterval:15000});
  const apiUp=h?.status==='UP', dbUp=h?.database?.status==='UP', cacheUp=h?.redis?.status==='UP';
  return (
    <aside style={{width:'210px',background:'var(--ink-2)',borderRight:'1px solid var(--line)',display:'flex',flexDirection:'column',flexShrink:0}}>
      <div style={{padding:'18px 16px 14px',borderBottom:'1px solid var(--line)'}}>
        <div style={{display:'flex',alignItems:'center',gap:'10px'}}>
          <div style={{width:'30px',height:'30px',background:'var(--cyan-dim)',border:'1px solid var(--cyan)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="1" width="5" height="5" stroke="#00e5cc" strokeWidth="1.2"/><rect x="8" y="1" width="5" height="5" stroke="#00e5cc" strokeWidth="1.2" opacity="0.5"/><rect x="1" y="8" width="5" height="5" stroke="#00e5cc" strokeWidth="1.2" opacity="0.5"/><rect x="8" y="8" width="5" height="5" stroke="#00e5cc" strokeWidth="1.2" opacity="0.25"/></svg>
          </div>
          <div>
            <div style={{fontFamily:'Space Mono, monospace',fontSize:'13px',fontWeight:700,color:'var(--bright)'}}>SpatialPath</div>
            <div style={{fontFamily:'Space Mono, monospace',fontSize:'9px',color:'var(--cyan)',letterSpacing:'.12em',textTransform:'uppercase'}}>DB · v1.0</div>
          </div>
        </div>
      </div>
      <nav style={{flex:1,padding:'10px 8px'}}>
        <div style={{fontFamily:'Space Mono, monospace',fontSize:'9px',fontWeight:700,letterSpacing:'.15em',textTransform:'uppercase',color:'var(--muted)',padding:'4px 8px 8px'}}>Navigation</div>
        {nav.map(({to,icon:Icon,label})=>(
          <NavLink key={to} to={to} className={({isActive})=>`nav-item${isActive?' active':''}`}>
            <Icon size={13} strokeWidth={2}/>{label}
          </NavLink>
        ))}
      </nav>
      <div style={{padding:'12px 16px',borderTop:'1px solid var(--line)'}}>
        <div style={{fontFamily:'Space Mono, monospace',fontSize:'9px',fontWeight:700,letterSpacing:'.12em',textTransform:'uppercase',color:'var(--muted)',marginBottom:'10px'}}>System</div>
        {([['API',apiUp,apiUp?'ONLINE':'···'],['DB',dbUp,dbUp?'PostGIS':'···'],['Cache',cacheUp,cacheUp?'Redis':'···']] as [string,boolean,string][]).map(([l,up,n])=>(
          <div key={l} style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'7px'}}>
            <span style={{fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--muted)'}}>{l}</span>
            <div style={{display:'flex',alignItems:'center',gap:'5px'}}><Dot up={up}/><span style={{fontFamily:'Space Mono, monospace',fontSize:'10px',color:up?'var(--green)':'var(--muted)'}}>{n}</span></div>
          </div>
        ))}
      </div>
    </aside>
  );
}
