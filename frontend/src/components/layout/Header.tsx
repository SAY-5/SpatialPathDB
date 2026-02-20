import { Activity } from 'lucide-react';
export default function Header() {
  const t = new Date().toISOString().replace('T',' ').slice(0,19);
  return (
    <header style={{height:'40px',background:'var(--ink)',borderBottom:'1px solid var(--line)',display:'flex',alignItems:'center',justifyContent:'space-between',padding:'0 20px',flexShrink:0}}>
      <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
        <Activity size={11} color="var(--muted)" />
        <span style={{fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--muted)',letterSpacing:'.06em'}}>Spatial Pathology Analytics Platform</span>
      </div>
      <span style={{fontFamily:'Space Mono, monospace',fontSize:'10px',color:'var(--muted)'}}>{t} UTC</span>
    </header>
  );
}
