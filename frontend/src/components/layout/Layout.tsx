import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
export default function Layout() {
  return (
    <div className="grid-bg" style={{display:'flex',height:'100vh',overflow:'hidden'}}>
      <Sidebar />
      <div style={{flex:1,display:'flex',flexDirection:'column',overflow:'hidden',minWidth:0}}>
        <Header />
        <main style={{flex:1,overflow:'auto',padding:'24px'}}><Outlet /></main>
      </div>
    </div>
  );
}
