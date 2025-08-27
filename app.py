"""
Multimodal Agentic AI Therapy System - Main Application
A localhost-based therapy assistant processing text, audio, and video inputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import time
import threading
import queue
from pathlib import Path
import base64

# Import our custom modules (will be created next)
try:
    from sentiment_analyzer import SentimentAnalyzer
    from audio_analyzer import AudioAnalyzer
    from visual_analyzer import VisualAnalyzer
    from multimodal_fusion import MultimodalFusion
    from therapy_agent import TherapyAgent
except ImportError as e:
    st.error(f"Missing module: {e}. Please ensure all components are available.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="AI Therapy Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MultimodalTherapyApp:
    def __init__(self):
        self.setup_directories()
        self.initialize_components()
        self.setup_session_state()
        
    def setup_directories(self):
        """Create necessary directories for data storage"""
        self.directories = {
            'sessions': Path('session_data'),
            'audio': Path('audio_data'),
            'video': Path('video_data'),
            'profiles': Path('multimodal_profiles'),
            'models': Path('models'),
            'logs': Path('logs')
        }
        
        for directory in self.directories.values():
            directory.mkdir(exist_ok=True)
    
    def initialize_components(self):
        """Initialize all analysis components"""
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            self.audio_analyzer = AudioAnalyzer()
            self.visual_analyzer = VisualAnalyzer()
            self.multimodal_fusion = MultimodalFusion()
            self.therapy_agent = TherapyAgent()
            
            st.success("✅ All components initialized successfully!")
        except Exception as e:
            st.error(f"❌ Component initialization failed: {e}")
            st.stop()
    
    def setup_session_state(self):
        """Initialize Streamlit session state variables"""
        defaults = {
            'session_id': None,
            'user_consents': {'text': False, 'audio': False, 'video': False},
            'chat_history': [],
            'analysis_results': [],
            'current_session_data': {},
            'audio_recording': False,
            'video_recording': False,
            'crisis_detected': False,
            'agent_active': True,
            'personalization_data': {},
            'intervention_history': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_privacy_consent(self):
        """Render privacy consent interface"""
        st.header("🔒 Privacy & Consent")
        st.markdown("""
        **Your Privacy is Our Priority**
        
        This AI therapy assistant processes your data locally on your device. 
        Please review and consent to the modalities you want to use:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📝 Text Analysis")
            st.markdown("- Chat conversations\n- Emotional sentiment\n- Crisis detection")
            st.session_state.user_consents['text'] = st.checkbox(
                "Enable Text Analysis", 
                value=st.session_state.user_consents['text'],
                key="text_consent"
            )
        
        with col2:
            st.subheader("🎤 Audio Analysis")
            st.markdown("- Voice tone analysis\n- Emotional prosody\n- Speech patterns")
            st.session_state.user_consents['audio'] = st.checkbox(
                "Enable Audio Analysis", 
                value=st.session_state.user_consents['audio'],
                key="audio_consent"
            )
        
        with col3:
            st.subheader("📹 Video Analysis")
            st.markdown("- Facial expressions\n- Body language\n- Micro-expressions")
            st.session_state.user_consents['video'] = st.checkbox(
                "Enable Video Analysis", 
                value=st.session_state.user_consents['video'],
                key="video_consent"
            )
        
        if any(st.session_state.user_consents.values()):
            if st.button("✅ Start Therapy Session", type="primary"):
                st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.rerun()
        else:
            st.warning("Please enable at least one modality to continue.")
    
    def render_multimodal_interface(self):
        """Render the main multimodal therapy interface"""
        # Header with session info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title("🧠 AI Therapy Assistant")
        with col2:
            st.metric("Session", st.session_state.session_id.split('_')[1])
        with col3:
            if st.button("🚪 End Session"):
                self.end_session()
                return
        
        # Create main layout
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            self.render_chat_interface()
        
        with right_col:
            self.render_analysis_dashboard()
        
        # Bottom row for multimodal controls
        self.render_multimodal_controls()
    
    def render_chat_interface(self):
        """Render the text chat interface"""
        st.subheader("💬 Therapy Chat")
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "analysis" in message:
                        with st.expander("📊 Analysis Details"):
                            st.json(message["analysis"])
        
        # Chat input
        user_input = st.chat_input("Share your thoughts...")
        if user_input:
            self.process_user_message(user_input)
    
    def render_analysis_dashboard(self):
        """Render real-time analysis dashboard"""
        st.subheader("📊 Real-time Analysis")
        
        if st.session_state.analysis_results:
            latest_analysis = st.session_state.analysis_results[-1]
            
            # Sentiment Meter
            sentiment_score = latest_analysis.get('sentiment_score', 0)
            st.metric(
                "Emotional State", 
                f"{latest_analysis.get('sentiment_label', 'Neutral')}", 
                f"{sentiment_score:.2f}"
            )
            
            # Crisis Alert
            if latest_analysis.get('crisis_risk', 0) > 0.7:
                st.error("🚨 Crisis Risk Detected")
                st.session_state.crisis_detected = True
            elif latest_analysis.get('crisis_risk', 0) > 0.5:
                st.warning("⚠️ Elevated Concern")
            else:
                st.success("✅ Normal Range")
            
            # Multimodal confidence
            if 'confidence_scores' in latest_analysis:
                st.subheader("🎯 Analysis Confidence")
                conf_scores = latest_analysis['confidence_scores']
                for modality, score in conf_scores.items():
                    st.progress(score, text=f"{modality.title()}: {score:.1%}")
        
        # Agent Recommendations
        self.render_agent_recommendations()
    
    def render_agent_recommendations(self):
        """Render AI agent recommendations"""
        st.subheader("🤖 AI Recommendations")
        
        if st.session_state.intervention_history:
            latest_intervention = st.session_state.intervention_history[-1]
            
            st.info(f"💡 {latest_intervention.get('recommendation', 'Continue sharing')}")
            
            if latest_intervention.get('technique'):
                technique = latest_intervention['technique']
                st.markdown(f"**Suggested Technique:** {technique['name']}")
                st.markdown(technique['description'])
                
                if st.button(f"Try {technique['name']}", key="try_technique"):
                    self.execute_technique(technique)
    
    def render_multimodal_controls(self):
        """Render controls for audio and video recording"""
        st.divider()
        st.subheader("🎛️ Multimodal Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.user_consents['audio']:
                if st.button("🎤 Record Audio" if not st.session_state.audio_recording else "⏹️ Stop Recording"):
                    self.toggle_audio_recording()
        
        with col2:
            if st.session_state.user_consents['video']:
                if st.button("📹 Start Video" if not st.session_state.video_recording else "⏹️ Stop Video"):
                    self.toggle_video_recording()
        
        with col3:
            st.metric("Audio", "🟢 Active" if st.session_state.audio_recording else "⚫ Inactive")
        
        with col4:
            st.metric("Video", "🟢 Active" if st.session_state.video_recording else "⚫ Inactive")
    
    def process_user_message(self, message):
        """Process user message through all available modalities"""
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })
        
        # Analyze message
        analysis_results = self.analyze_multimodal_input(text=message)
        
        # Get AI agent response
        agent_response = self.therapy_agent.generate_response(
            message, 
            analysis_results, 
            st.session_state.chat_history
        )
        
        # Add agent response to chat
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": agent_response['message'],
            "analysis": analysis_results,
            "timestamp": datetime.now()
        })
        
        # Store analysis results
        st.session_state.analysis_results.append(analysis_results)
        
        # Check for interventions
        if agent_response.get('intervention'):
            st.session_state.intervention_history.append(agent_response['intervention'])
        
        # Crisis handling
        if analysis_results.get('crisis_risk', 0) > 0.8:
            self.handle_crisis_situation(analysis_results)
        
        st.rerun()
    
    def analyze_multimodal_input(self, text=None, audio_data=None, video_data=None):
        """Analyze input across all consented modalities"""
        results = {}
        
        # Text analysis (if consented)
        if text and st.session_state.user_consents['text']:
            text_results = self.sentiment_analyzer.analyze(text)
            results['text'] = text_results
        
        # Audio analysis (if consented and available)
        if audio_data and st.session_state.user_consents['audio']:
            audio_results = self.audio_analyzer.analyze(audio_data)
            results['audio'] = audio_results
        
        # Video analysis (if consented and available)
        if video_data and st.session_state.user_consents['video']:
            video_results = self.visual_analyzer.analyze(video_data)
            results['visual'] = video_results
        
        # Multimodal fusion
        if len(results) > 1:
            fused_results = self.multimodal_fusion.fuse_analyses(results)
            results['fused'] = fused_results
            
            # Use fused results as primary analysis
            primary_analysis = fused_results
        elif results:
            # Use single modality results
            primary_analysis = list(results.values())[0]
        else:
            # Fallback
            primary_analysis = {'sentiment_score': 0, 'sentiment_label': 'Neutral'}
        
        # Add metadata
        results.update(primary_analysis)
        results['timestamp'] = datetime.now()
        results['modalities_used'] = list(results.keys())
        
        return results
    
    def toggle_audio_recording(self):
        """Toggle audio recording state"""
        st.session_state.audio_recording = not st.session_state.audio_recording
        
        if st.session_state.audio_recording:
            st.success("🎤 Audio recording started")
            # Start audio recording in background thread
            self.start_audio_recording_thread()
        else:
            st.success("⏹️ Audio recording stopped")
    
    def toggle_video_recording(self):
        """Toggle video recording state"""
        st.session_state.video_recording = not st.session_state.video_recording
        
        if st.session_state.video_recording:
            st.success("📹 Video recording started")
            # Start video recording in background thread
            self.start_video_recording_thread()
        else:
            st.success("⏹️ Video recording stopped")
    
    def start_audio_recording_thread(self):
        """Start audio recording in background thread"""
        # Implementation placeholder - will connect to audio_analyzer
        pass
    
    def start_video_recording_thread(self):
        """Start video recording in background thread"""
        # Implementation placeholder - will connect to visual_analyzer
        pass
    
    def execute_technique(self, technique):
        """Execute a therapeutic technique"""
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Let's try the {technique['name']} technique together.",
            "timestamp": datetime.now()
        })
        
        # Execute technique steps
        for step in technique.get('steps', []):
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": step,
                "timestamp": datetime.now()
            })
        
        st.rerun()
    
    def handle_crisis_situation(self, analysis_results):
        """Handle crisis situation with appropriate escalation"""
        st.error("🚨 Crisis Situation Detected")
        
        crisis_response = {
            "role": "assistant",
            "content": """I'm concerned about what you're sharing. Your safety is important to me. 
            
If you're having thoughts of harming yourself or others, please reach out for immediate help:
- Emergency Services: 911 (US) or your local emergency number
- Crisis Hotline: 988 (US Suicide & Crisis Lifeline)
- Text HOME to 741741 (Crisis Text Line)

Would you like me to help you find local crisis resources?""",
            "timestamp": datetime.now(),
            "crisis_alert": True
        }
        
        st.session_state.chat_history.append(crisis_response)
        st.session_state.crisis_detected = True
    
    def end_session(self):
        """End current therapy session and save data"""
        session_data = {
            'session_id': st.session_state.session_id,
            'start_time': st.session_state.get('session_start', datetime.now()),
            'end_time': datetime.now(),
            'chat_history': st.session_state.chat_history,
            'analysis_results': st.session_state.analysis_results,
            'intervention_history': st.session_state.intervention_history,
            'crisis_detected': st.session_state.crisis_detected,
            'modalities_used': st.session_state.user_consents
        }
        
        # Save session data
        session_file = self.directories['sessions'] / f"{st.session_state.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, default=str, indent=2)
        
        st.success(f"Session saved: {session_file}")
        
        # Reset session state
        for key in ['session_id', 'chat_history', 'analysis_results', 
                   'intervention_history', 'crisis_detected']:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        
        st.rerun()
    
    def run(self):
        """Main application runner"""
        # Sidebar with information
        with st.sidebar:
            st.title("ℹ️ System Info")
            st.markdown("**Multimodal AI Therapy Assistant**")
            st.markdown("Local processing for maximum privacy")
            
            enabled_modalities = [k for k, v in st.session_state.user_consents.items() if v]
            st.markdown(f"**Active Modalities:** {', '.join(enabled_modalities) if enabled_modalities else 'None'}")
            
            if st.session_state.session_id:
                st.markdown(f"**Session ID:** {st.session_state.session_id}")
                st.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
                
                if st.button("📊 View Analytics"):
                    st.session_state.show_analytics = True
        
        # Main interface logic
        if not st.session_state.session_id:
            self.render_privacy_consent()
        else:
            self.render_multimodal_interface()
        
        # Footer
        st.divider()
        st.markdown("🔒 **Privacy Notice:** All processing occurs locally. No data leaves your device.")


def main():
    """Main application entry point"""
    try:
        app = MultimodalTherapyApp()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.stop()


if __name__ == "__main__":
    main()