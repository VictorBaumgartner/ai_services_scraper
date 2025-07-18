import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse, urljoin
import time
import re
from urllib.robotparser import RobotFileParser
import urllib3
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Download NLTK data (run once if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Category mapping table
CATEGORY_MAPPING = {
    "Entreprises": {
        "Gestion et Administration": "Productivity",
        "Marketing et Ventes": "Creative",
        "Chaîne d'Approvisionnement et Logistique": "Productivity",
        "Production et Industrie": "Development",
        "Service Client": "Productivity",
        "Recherche et Développement": "Development",
        "Sécurité et Cybersécurité": "Security",
        "Secteurs Spécifiques": "Productivity"
    },
    "Particuliers": {
        "Vie Quotidienne et Productivité": "Productivity",
        "Divertissement": "Media",
        "Santé et Bien-être": "Productivity",
        "Finances Personnelles": "Productivity",
        "Éducation et Apprentissage": "Productivity",
        "Maison Intelligente": "Development",
        "Mobilité et Transport": "Development",
        "Communication": "Media",
        "Shopping": "Productivity"
    },
    "Cas Émergents": {
        "IA émotionnelle": "Creative",
        "Durabilité": "Development",
        "Jumeaux numériques": "Development",
        "IA juridique": "Security",
        "Accessibilité": "Development",
        "Simulation/formation": "Development"
    }
}

# Valid service types
SERVICE_TYPES = [
    "computer-vision", "nlp", "speech-recognition", "tts", "conversational-ai",
    "recommendation", "document-intelligence", "rpa", "predictive-analytics",
    "anomaly-detection", "optimization", "autonomous-agents", "rules-ai",
    "drug-discovery", "medical-diagnosis", "bioinformatics", "physics-chemistry",
    "weather-climate", "digital-twins", "crm", "hr-talent", "finance",
    "marketing", "legal", "supply-chain", "text-generation", "image-generation",
    "music-composition", "video-generation", "code-generation", "cybersecurity",
    "compliance", "risk-management", "xai", "bias-fairness", "model-training",
    "inference", "automl", "data-labeling", "embedding-search"
]

# Comprehensive master feature list with extreme granularity
MASTER_FEATURES = [
    # === TEXT PROCESSING & NLP ===
    "Text Generation", "Text Summarization", "Text Paraphrasing", "Text Rewriting",
    "Text Expansion", "Text Compression", "Text Simplification", "Text Complexity Analysis",
    "Grammar Correction", "Spell Check", "Punctuation Correction", "Style Transfer",
    "Tone Adjustment", "Formality Control", "Readability Optimization", "Clarity Enhancement",
    "Sentiment Analysis", "Emotion Detection", "Mood Analysis", "Intent Recognition",
    "Sarcasm Detection", "Irony Detection", "Subtext Analysis", "Implicit Meaning Extraction",
    "Named Entity Recognition", "Entity Linking", "Relation Extraction", "Event Extraction",
    "Temporal Information Extraction", "Spatial Information Extraction", "Numerical Information Extraction",
    "Keyword Extraction", "Keyphrase Extraction", "Topic Modeling", "Theme Identification",
    "Semantic Similarity", "Textual Entailment", "Text Classification", "Document Categorization",
    "Genre Classification", "Authorship Attribution", "Plagiarism Detection", "Content Originality Check",
    "Language Detection", "Language Identification", "Dialect Recognition", "Accent Classification",
    "Multi-language Support", "Cross-lingual Transfer", "Machine Translation", "Neural Machine Translation",
    "Real-time Translation", "Contextual Translation", "Domain-specific Translation", "Literary Translation",
    "Technical Translation", "Legal Translation", "Medical Translation", "Scientific Translation",
    "Question Answering", "Reading Comprehension", "Information Retrieval", "Fact Checking",
    "Knowledge Graph Construction", "Ontology Building", "Concept Mapping", "Semantic Parsing",
    "Coreference Resolution", "Anaphora Resolution", "Discourse Analysis", "Coherence Analysis",
    "Text Coherence Scoring", "Narrative Structure Analysis", "Story Arc Detection", "Character Analysis",
    "Plot Analysis", "Dialogue Analysis", "Monologue Analysis", "Conversational Analysis",
    "Argumentation Mining", "Claim Detection", "Evidence Extraction", "Bias Detection in Text",
    "Fake News Detection", "Misinformation Detection", "Propaganda Detection", "Manipulation Detection",
    "Hate Speech Detection", "Offensive Language Detection", "Toxicity Detection", "Cyberbullying Detection",
    "Content Moderation", "NSFW Text Detection", "Adult Content Detection", "Violence Detection in Text",
    "Threat Detection", "Harassment Detection", "Discrimination Detection", "Ethical Content Analysis",
    
    # === SPEECH & AUDIO ===
    "Text-to-Speech", "Speech-to-Text", "Voice Synthesis", "Voice Cloning", "Voice Conversion",
    "Voice Morphing", "Voice Aging", "Voice Gender Conversion", "Voice Accent Modification",
    "Voice Emotion Synthesis", "Expressive Speech Synthesis", "Prosody Control", "Intonation Control",
    "Rhythm Control", "Pace Control", "Pitch Control", "Volume Control", "Timbre Control",
    "Speech Recognition", "Real-time Speech Recognition", "Offline Speech Recognition",
    "Continuous Speech Recognition", "Isolated Word Recognition", "Speaker Recognition",
    "Speaker Identification", "Speaker Verification", "Speaker Diarization", "Speaker Segmentation",
    "Voice Activity Detection", "Speech Endpoint Detection", "Silence Detection", "Noise Reduction",
    "Audio Enhancement", "Audio Denoising", "Audio Restoration", "Audio Equalization",
    "Audio Compression", "Audio Normalization", "Audio Amplification", "Audio Filtering",
    "Echo Cancellation", "Reverb Removal", "Feedback Suppression", "Acoustic Echo Cancellation",
    "Audio Transcription", "Meeting Transcription", "Interview Transcription", "Lecture Transcription",
    "Phoneme Recognition", "Phonetic Transcription", "Pronunciation Analysis", "Accent Analysis",
    "Dialect Recognition", "Language Identification from Audio", "Multilingual Speech Recognition",
    "Code-switching Detection", "Emotion Recognition from Speech", "Stress Detection from Voice",
    "Fatigue Detection from Voice", "Health Monitoring from Voice", "Mood Detection from Voice",
    "Personality Assessment from Voice", "Age Estimation from Voice", "Gender Recognition from Voice",
    "Audio Synthesis", "Sound Effect Generation", "Music Generation", "Melody Generation",
    "Rhythm Generation", "Harmony Generation", "Chord Progression", "Beat Detection",
    "Tempo Analysis", "Key Detection", "Genre Classification", "Mood Classification from Audio",
    "Instrument Recognition", "Vocal Separation", "Audio Source Separation", "Stem Separation",
    "Karaoke Mode", "Vocal Removal", "Instrumental Extraction", "Audio Mixing", "Audio Mastering",
    
    # === COMPUTER VISION & IMAGE PROCESSING ===
    "Image Generation", "Image Synthesis", "Image Creation", "AI Art Generation", "Digital Art Creation",
    "Photorealistic Image Generation", "Artistic Style Transfer", "Neural Style Transfer",
    "Image Style Conversion", "Image Stylization", "Image Painting", "Image Sketching",
    "Image Recognition", "Object Detection", "Object Recognition", "Object Classification",
    "Object Localization", "Object Segmentation", "Instance Segmentation", "Semantic Segmentation",
    "Panoptic Segmentation", "Scene Understanding", "Scene Recognition", "Scene Classification",
    "Scene Segmentation", "Scene Parsing", "Scene Reconstruction", "3D Scene Understanding",
    "Facial Recognition", "Face Detection", "Face Identification", "Face Verification",
    "Face Matching", "Facial Landmark Detection", "Facial Expression Recognition",
    "Emotion Recognition from Images", "Age Estimation from Images", "Gender Recognition from Images",
    "Ethnicity Recognition", "Facial Attribute Analysis", "Beauty Score Prediction",
    "Makeup Simulation", "Hairstyle Simulation", "Aging Simulation", "Face Morphing",
    "Face Swapping", "Face Reenactment", "Face Animation", "Lip Sync", "Eye Gaze Tracking",
    "Head Pose Estimation", "Gesture Recognition", "Hand Tracking", "Body Pose Estimation",
    "Human Activity Recognition", "Action Recognition", "Behavior Analysis", "Crowd Analysis",
    "People Counting", "Person Re-identification", "Pedestrian Detection", "Vehicle Detection",
    "License Plate Recognition", "Traffic Sign Recognition", "Road Scene Understanding",
    "Autonomous Driving Vision", "Medical Image Analysis", "X-ray Analysis", "CT Scan Analysis",
    "MRI Analysis", "Ultrasound Analysis", "Pathology Image Analysis", "Skin Lesion Detection",
    "Retinal Image Analysis", "Dental X-ray Analysis", "Microscopy Image Analysis",
    "Satellite Image Analysis", "Aerial Image Analysis", "Geographic Information Extraction",
    "Land Use Classification", "Vegetation Analysis", "Urban Planning Analysis",
    "Image Enhancement", "Image Restoration", "Image Denoising", "Image Deblurring",
    "Image Sharpening", "Image Super-resolution", "Image Upscaling", "Image Downscaling",
    "Image Colorization", "Image Decolorization", "Color Transfer", "Color Correction",
    "White Balance Correction", "Exposure Correction", "Contrast Enhancement", "Brightness Adjustment",
    "Saturation Adjustment", "Hue Adjustment", "Gamma Correction", "Histogram Equalization",
    "Background Removal", "Background Replacement", "Background Blurring", "Foreground Extraction",
    "Green Screen Removal", "Chroma Key", "Alpha Matting", "Image Compositing",
    "Image Blending", "Image Stitching", "Panorama Creation", "HDR Image Creation",
    "Depth Estimation", "Stereo Vision", "3D Reconstruction", "Point Cloud Generation",
    "Mesh Generation", "Texture Mapping", "3D Model Creation", "SLAM", "Visual Odometry",
    "Camera Calibration", "Lens Distortion Correction", "Perspective Correction",
    "Geometric Transformation", "Image Warping", "Image Morphing", "Image Interpolation",
    "Optical Character Recognition", "Text Detection in Images", "Text Recognition",
    "Document Scanning", "Receipt Scanning", "Business Card Scanning", "Handwriting Recognition",
    "Signature Recognition", "Barcode Detection", "QR Code Detection", "Logo Detection",
    "Brand Recognition", "Product Recognition", "Fashion Recognition", "Food Recognition",
    "Plant Recognition", "Animal Recognition", "Landmark Recognition", "Artwork Recognition",
    "Reverse Image Search", "Visual Search", "Content-based Image Retrieval",
    "Image Similarity Search", "Duplicate Image Detection", "Near-duplicate Detection",
    "Image Forensics", "Deepfake Detection", "Image Authenticity Verification",
    "Image Tampering Detection", "Copy-move Detection", "Splicing Detection",
    "NSFW Image Detection", "Adult Content Detection", "Violence Detection in Images",
    "Inappropriate Content Detection", "Content Safety Analysis", "Image Content Moderation",
    
    # === VIDEO PROCESSING & ANALYSIS ===
    "Video Generation", "Video Synthesis", "Video Creation", "AI Video Generation",
    "Text-to-Video", "Image-to-Video", "Video-to-Video", "Video Style Transfer",
    "Video Colorization", "Video Denoising", "Video Super-resolution", "Video Upscaling",
    "Video Stabilization", "Video Deblurring", "Video Enhancement", "Video Restoration",
    "Frame Interpolation", "Slow Motion Generation", "Time-lapse Creation", "Speed Adjustment",
    "Video Compression", "Video Encoding", "Video Transcoding", "Format Conversion",
    "Video Editing", "Auto-editing", "Smart Editing", "Intelligent Editing",
    "Scene Detection", "Shot Detection", "Cut Detection", "Transition Detection",
    "Temporal Segmentation", "Video Summarization", "Highlight Detection", "Key Frame Extraction",
    "Video Thumbnails", "Video Previews", "Video Trailers", "Video Montage",
    "Auto-trimming", "Content-aware Trimming", "Silence Detection", "Dead Space Removal",
    "Filler Word Removal", "Uh Detection", "Pause Detection", "Breath Detection",
    "Auto-captioning", "Subtitle Generation", "Closed Caption Creation", "Multi-language Subtitles",
    "Subtitle Translation", "Subtitle Synchronization", "Subtitle Styling", "Subtitle Positioning",
    "Video Repurposing", "Content Adaptation", "Aspect Ratio Conversion", "Vertical Video Creation",
    "Horizontal Video Creation", "Square Video Creation", "Social Media Optimization",
    "Platform-specific Formatting", "Multi-platform Distribution", "Cross-platform Adaptation",
    "Stock Footage Integration", "B-Roll Insertion", "AI B-Roll Generation", "Footage Matching",
    "Scene Matching", "Visual Continuity", "Color Matching", "Style Consistency",
    "Branded Templates", "Brand Guidelines Enforcement", "Logo Insertion", "Watermark Addition",
    "Brand Color Application", "Corporate Style Transfer", "Template Customization",
    "Video Analytics", "Engagement Analytics", "View Duration Analysis", "Drop-off Analysis",
    "Attention Mapping", "Heatmap Generation", "Viewer Behavior Analysis", "Performance Metrics",
    "Video SEO", "Metadata Generation", "Tag Suggestion", "Title Optimization",
    "Description Generation", "Thumbnail Optimization", "Video Search Optimization",
    "Motion Detection", "Object Tracking", "Multi-object Tracking", "Person Tracking",
    "Vehicle Tracking", "Animal Tracking", "Sports Analysis", "Player Tracking",
    "Ball Tracking", "Game Analysis", "Performance Analysis", "Tactical Analysis",
    "Surveillance Analysis", "Anomaly Detection in Video", "Intrusion Detection",
    "Crowd Monitoring", "Traffic Analysis", "Behavior Analysis", "Activity Recognition",
    "Action Detection", "Violence Detection", "Fight Detection", "Accident Detection",
    "Medical Video Analysis", "Surgical Video Analysis", "Endoscopy Analysis",
    "Radiology Video Analysis", "Diagnostic Video Analysis", "Therapeutic Video Analysis",
    "Educational Video Analysis", "Training Video Analysis", "Instructional Video Analysis",
    "E-learning Video Analysis", "Presentation Analysis", "Lecture Analysis",
    "Video Accessibility", "Audio Description Generation", "Sign Language Recognition",
    "Sign Language Translation", "Visual Description", "Scene Description",
    "NSFW Video Detection", "Adult Content Detection", "Violence Detection in Video",
    "Inappropriate Content Detection", "Content Safety Analysis", "Video Content Moderation",
    
    # === CODE GENERATION & DEVELOPMENT ===
    "Code Generation", "Code Synthesis", "Code Creation", "Programming Assistance",
    "Code Completion", "Code Suggestion", "Code Prediction", "Code Autocomplete",
    "Code Debugging", "Bug Detection", "Error Identification", "Exception Handling",
    "Code Review", "Code Quality Analysis", "Code Smell Detection", "Anti-pattern Detection",
    "Code Optimization", "Performance Optimization", "Memory Optimization", "Speed Optimization",
    "Code Refactoring", "Code Restructuring", "Code Simplification", "Code Modernization",
    "Legacy Code Migration", "Framework Migration", "Language Migration", "Platform Migration",
    "Code Translation", "Cross-language Translation", "Transpilation", "Code Conversion",
    "Syntax Highlighting", "Code Formatting", "Code Beautification", "Code Styling",
    "Code Documentation", "Comment Generation", "API Documentation", "README Generation",
    "Code Explanation", "Code Understanding", "Algorithm Explanation", "Logic Explanation",
    "Code Testing", "Unit Test Generation", "Integration Test Generation", "Test Case Creation",
    "Test Data Generation", "Mock Data Creation", "Test Automation", "Test Coverage Analysis",
    "Code Security Analysis", "Vulnerability Detection", "Security Audit", "Penetration Testing",
    "Code Injection Detection", "SQL Injection Detection", "XSS Detection", "CSRF Detection",
    "Code Deployment", "CI/CD Pipeline", "Build Automation", "Deployment Automation",
    "Configuration Management", "Environment Management", "Container Management", "Orchestration",
    "Code Monitoring", "Performance Monitoring", "Error Monitoring", "Log Analysis",
    "Metrics Collection", "Alerting", "Notification", "Incident Response",
    "Version Control", "Git Integration", "Branch Management", "Merge Conflict Resolution",
    "Code Collaboration", "Pair Programming", "Code Review Automation", "Team Collaboration",
    "Project Management", "Task Management", "Issue Tracking", "Bug Tracking",
    "Feature Request Management", "Requirement Analysis", "User Story Creation",
    "Database Code Generation", "SQL Generation", "Query Optimization", "Database Schema Design",
    "ORM Code Generation", "Database Migration", "Data Modeling", "Entity Relationship Modeling",
    "API Code Generation", "REST API Generation", "GraphQL API Generation", "RPC API Generation",
    "Web Service Creation", "Microservice Architecture", "Service Mesh", "API Gateway",
    "Frontend Code Generation", "UI Component Generation", "React Component Generation",
    "Vue Component Generation", "Angular Component Generation", "Web Component Generation",
    "CSS Generation", "HTML Generation", "JavaScript Generation", "TypeScript Generation",
    "Mobile App Code Generation", "iOS App Generation", "Android App Generation",
    "Cross-platform Development", "Flutter Code Generation", "React Native Code Generation",
    "Game Development Code", "Unity Script Generation", "Unreal Script Generation",
    "Game Logic Generation", "AI Behavior Generation", "Procedural Generation",
    "Machine Learning Code", "ML Model Generation", "Neural Network Code", "Deep Learning Code",
    "Data Science Code", "Data Analysis Code", "Statistical Analysis Code", "Visualization Code",
    "DevOps Code", "Infrastructure as Code", "Terraform Code", "Ansible Code", "Docker Code",
    "Kubernetes YAML", "Helm Charts", "Cloud Formation", "ARM Templates",
    
    # === CONVERSATIONAL AI & CHATBOTS ===
    "Chatbot Functionality", "Conversational AI", "Virtual Assistant", "Digital Assistant",
    "Conversational Interface", "Natural Language Interface", "Voice Interface", "Chat Interface",
    "Conversational Context Retention", "Context Awareness", "Memory Management", "Session Management",
    "Conversation History", "Long-term Memory", "Short-term Memory", "Episodic Memory",
    "Intent Recognition", "Intent Classification", "Intent Extraction", "Goal Recognition",
    "Task Recognition", "Action Recognition", "Command Recognition", "Request Recognition",
    "Entity Recognition", "Slot Filling", "Parameter Extraction", "Information Extraction",
    "Dialogue Management", "Conversation Flow", "State Management", "Turn Management",
    "Dialogue Strategy", "Response Selection", "Response Generation", "Response Ranking",
    "Multi-turn Conversation", "Context Switching", "Topic Switching", "Conversation Repair",
    "Clarification Requests", "Confirmation Requests", "Disambiguation", "Error Recovery",
    "Personality Modeling", "Persona Development", "Character Consistency", "Emotional Intelligence",
    "Empathy Modeling", "Social Intelligence", "Cultural Awareness", "Behavioral Modeling",
    "Conversation Analytics", "Dialogue Analysis", "Conversation Mining", "User Behavior Analysis",
    "Engagement Metrics", "Satisfaction Metrics", "Conversation Quality", "Response Quality",
    "Conversation Testing", "Dialogue Testing", "User Testing", "A/B Testing",
    "Conversation Optimization", "Response Optimization", "Flow Optimization", "Performance Tuning",
    "Multi-language Conversations", "Cross-lingual Dialogue", "Translation in Conversation",
    "Language Switching", "Code-switching", "Multilingual Support", "Localization",
    "Voice Conversations", "Speech-based Dialogue", "Voice Commands", "Voice Responses",
    "Multimodal Conversations", "Text and Voice", "Visual Conversations", "Gesture-based Interaction",
    "Customer Service Chatbots", "Support Chatbots", "Help Desk Automation", "FAQ Bots",
    "Troubleshooting Bots", "Technical Support Bots", "Order Processing Bots", "Booking Bots",
    "Sales Chatbots", "Lead Generation Bots", "Qualification Bots", "Product Recommendation Bots",
    "E-commerce Bots", "Shopping Assistants", "Personal Shopping Bots", "Style Advisors",
    "Health Chatbots", "Medical Assistance Bots", "Symptom Checkers", "Medication Reminders",
    "Therapy Bots", "Mental Health Support", "Wellness Coaching", "Fitness Coaching",
    "Educational Chatbots", "Tutoring Bots", "Learning Assistants", "Study Companions",
    "Language Learning Bots", "Skill Development Bots", "Training Bots", "Coaching Bots",
    "Entertainment Chatbots", "Gaming Bots", "Storytelling Bots", "Companion Bots",
    "Social Chatbots", "Friend Bots", "Relationship Bots", "Dating Bots",
    
    # === RECOMMENDATION SYSTEMS ===
    "Recommendation Engine", "Personalization", "Content Recommendation", "Product Recommendation",
    "Collaborative Filtering", "Content-based Filtering", "Hybrid Filtering", "Matrix Factorization",
    "Deep Learning Recommendations", "Neural Collaborative Filtering", "Autoencoder Recommendations",
    "Reinforcement Learning Recommendations", "Multi-armed Bandit", "Contextual Bandits",
    "Real-time Recommendations", "Batch Recommendations", "Online Learning", "Incremental Learning",
    "Cold Start Problem", "New User Recommendations", "New Item Recommendations", "Bootstrap Recommendations",
    "Diversity in Recommendations", "Novelty in Recommendations", "Serendipity", "Exploration vs Exploitation",
    "Popularity-based Recommendations", "Trending Recommendations", "Seasonal Recommendations",
    "Location-based Recommendations", "Geo-spatial Recommendations", "Local Recommendations",
    "Time-aware Recommendations", "Temporal Recommendations", "Sequential Recommendations",
    "Session-based Recommendations", "Next-item Prediction", "Sequence Prediction",
    "Multi-criteria Recommendations", "Multi-objective Recommendations", "Preference Learning",
    "Implicit Feedback", "Explicit Feedback", "Rating Prediction", "Preference Prediction",
    "Cross-domain Recommendations", "Transfer Learning", "Domain Adaptation", "Multi-domain",
    "Group Recommendations", "Social Recommendations", "Friend Recommendations", "Influencer Recommendations",
    "Explanation Generation", "Recommendation Explanation", "Transparent Recommendations",
    "Recommendation Fairness", "Bias Mitigation", "Fairness Constraints", "Equitable Recommendations",
    "A/B Testing for Recommendations", "Recommendation Evaluation", "Offline Evaluation", "Online Evaluation",
    "Click-through Rate Prediction", "Conversion Rate Prediction", "Engagement Prediction",
    "Churn Prediction", "Retention Prediction", "Lifetime Value Prediction", "Revenue Optimization",
    "Music Recommendations", "Movie Recommendations", "Book Recommendations", "News Recommendations",
    "Video Recommendations", "Podcast Recommendations", "Article Recommendations", "Blog Recommendations",
    "E-commerce Recommendations", "Shopping Recommendations", "Fashion Recommendations", "Food Recommendations",
    "Restaurant Recommendations", "Travel Recommendations", "Hotel Recommendations", "Flight Recommendations",
    "Job Recommendations", "Career Recommendations", "Skill Recommendations", "Course Recommendations",
    "Learning Path Recommendations", "Content Curation", "Playlist Generation", "Collection Building",
    
    # === PREDICTIVE ANALYTICS & FORECASTING ===
    "Predictive Analytics", "Forecasting", "Time Series Prediction", "Trend Analysis",
    "Seasonal Forecasting", "Cyclical Forecasting", "Demand Forecasting", "Sales Forecasting",
    "Revenue Forecasting", "Market Forecasting", "Economic Forecasting", "Financial Forecasting",
    "Stock Price Prediction", "Cryptocurrency Prediction", "Commodity Price Prediction",
    "Risk Prediction", "Credit Risk Assessment", "Default Prediction", "Fraud Detection",
    "Anomaly Detection", "Outlier Detection", "Change Point Detection", "Drift Detection",
    "Pattern Recognition", "Sequence Analysis", "Event Prediction", "Outcome Prediction",
    "Behavioral Prediction", "Customer Behavior", "User Behavior", "Purchase Prediction",
    "Churn Prediction", "Retention Prediction", "Lifetime Value Prediction", "Engagement Prediction",
    "Health Outcome Prediction", "Disease Prediction", "Epidemic Forecasting", "Medical Diagnosis",
    "Treatment Outcome Prediction", "Drug Discovery", "Clinical Trial Prediction", "Biomarker Discovery",
    "Weather Forecasting", "Climate Prediction", "Environmental Forecasting", "Natural Disaster Prediction",
    "Earthquake Prediction", "Flood Prediction", "Hurricane Prediction", "Wildfire Prediction",
    "Traffic Prediction", "Transportation Forecasting", "Route Optimization", "Demand Response",
    "Energy Forecasting", "Load Forecasting", "Renewable Energy Prediction", "Grid Optimization",
    "Supply Chain Forecasting", "Inventory Prediction", "Logistics Optimization", "Delivery Prediction",
    "Manufacturing Prediction", "Production Forecasting", "Quality Prediction", "Yield Prediction",
    "Maintenance Prediction", "Equipment Failure", "Predictive Maintenance", "Condition Monitoring",
    "Performance Prediction", "Efficiency Forecasting", "Optimization Prediction", "Resource Planning",
    "Capacity Planning", "Workforce Planning", "Scheduling Optimization", "Resource Allocation",
    "Budget Forecasting", "Cost Prediction", "Price Optimization", "Profitability Analysis",
    "ROI Prediction", "Investment Analysis", "Portfolio Optimization", "Risk Management",
    "Stress Testing", "Scenario Analysis", "Sensitivity Analysis", "Monte Carlo Simulation",
    "Stochastic Modeling", "Probabilistic Forecasting", "Uncertainty Quantification", "Confidence Intervals",
    "Ensemble Methods", "Model Averaging", "Consensus Forecasting", "Hybrid Models",
    "Real-time Prediction", "Streaming Analytics", "Online Learning", "Adaptive Models",
    "Causal Inference", "Causal Modeling", "Intervention Analysis", "Counterfactual Analysis",
    "Treatment Effect Estimation", "A/B Test Analysis", "Experimental Design", "Statistical Testing",
    
    # === DOCUMENT INTELLIGENCE & PROCESSING ===
    "Document Processing", "Document Understanding", "Document Analysis", "Document Intelligence",
    "Document Parsing", "Document Structure Analysis", "Layout Analysis", "Page Layout Detection",
    "Text Extraction", "Data Extraction", "Information Extraction", "Content Extraction",
    "Form Processing", "Form Recognition", "Form Extraction", "Form Understanding",
    "Table Extraction", "Table Recognition", "Table Understanding", "Tabular Data Extraction",
    "Invoice Processing", "Invoice Recognition", "Invoice Data Extraction", "Invoice Validation",
    "Receipt Processing", "Receipt Recognition", "Receipt Data Extraction", "Expense Processing",
    "Contract Analysis", "Contract Understanding", "Contract Extraction", "Legal Document Analysis",
    "Clause Extraction", "Term Extraction", "Obligation Extraction", "Risk Analysis",
    "Resume Parsing", "CV Analysis", "Job Application Processing", "Candidate Screening",
    "Skill Extraction", "Experience Extraction", "Education Extraction", "Qualification Analysis",
    "Medical Record Processing", "Clinical Document Analysis", "Patient Record Extraction",
    "Medical Report Analysis", "Lab Result Processing", "Diagnosis Extraction",
    "Financial Document Processing", "Bank Statement Analysis", "Transaction Processing",
    "Tax Document Processing", "Audit Document Analysis", "Compliance Document Processing",
    "Academic Paper Processing", "Research Document Analysis", "Citation Extraction",
    "Bibliography Processing", "Literature Review", "Academic Writing Analysis",
    "News Article Processing", "Press Release Analysis", "Media Document Processing",
    "Social Media Document Processing", "Web Page Analysis", "Blog Post Processing",
    "Email Processing", "Email Analysis", "Email Classification", "Email Extraction",
    "Spam Detection", "Phishing Detection", "Email Security", "Email Sentiment Analysis",
    "PDF Processing", "PDF Text Extraction", "PDF Image Extraction", "PDF Parsing",
    "Word Document Processing", "Excel Processing", "PowerPoint Processing", "Presentation Analysis",
    "Slide Analysis", "Chart Extraction", "Graph Extraction", "Diagram Analysis",
    "Handwritten Document Processing", "Handwriting Recognition", "Signature Recognition",
    "Handwritten Form Processing", "Cursive Recognition", "Print Recognition",
    "Multi-language Document Processing", "Cross-lingual Document Analysis", "Translation Memory",
    "Terminology Extraction", "Glossary Creation", "Dictionary Building", "Knowledge Base Creation",
    "Document Classification", "Document Categorization", "Document Clustering", "Document Similarity",
    "Document Retrieval", "Document Search", "Document Indexing", "Document Ranking",
    "Document Summarization", "Document Abstraction", "Key Point Extraction", "Executive Summary",
    "Document Comparison", "Document Diff", "Version Comparison", "Change Detection",
    "Document Validation", "Document Verification", "Document Authenticity", "Document Compliance",
    "Document Workflow", "Document Routing", "Document Approval", "Document Review",
    "Document Collaboration", "Document Sharing", "Document Comments", "Document Annotations",
    "Document Security", "Document Encryption", "Document Access Control", "Document Permissions",
    "Document Audit", "Document Tracking", "Document History", "Document Versioning",
    "Document Archiving", "Document Retention", "Document Disposal", "Document Lifecycle",
    
    # === ROBOTIC PROCESS AUTOMATION (RPA) ===
    "Process Automation", "Task Automation", "Workflow Automation", "Business Process Automation",
    "Robotic Process Automation", "Intelligent Automation", "Cognitive Automation", "Hyperautomation",
    "Screen Scraping", "Web Scraping", "Data Scraping", "Information Scraping",
    "UI Automation", "GUI Automation", "Desktop Automation", "Application Automation",
    "Web Automation", "Browser Automation", "Mobile Automation", "API Automation",
    "Form Automation", "Data Entry Automation", "Report Generation", "Report Automation",
    "Email Automation", "Communication Automation", "Notification Automation", "Alert Automation",
    "File Processing", "File Manipulation", "File Transfer", "File Synchronization",
    "Database Automation", "SQL Automation", "Data Migration", "Data Transformation",
    "ETL Automation", "Data Pipeline", "Data Processing", "Batch Processing",
    "Scheduled Tasks", "Cron Jobs", "Timer-based Automation", "Event-driven Automation",
    "Rule-based Automation", "Decision Automation", "Logic Automation", "Conditional Processing",
    "Exception Handling", "Error Recovery", "Retry Logic", "Failover Mechanisms",
    "Process Monitoring", "Performance Monitoring", "Health Checks", "Status Reporting"


    # === CONTINUATION OF MASTER_FEATURES LIST ===
    "Audit Trail", "Compliance Reporting", "Governance", "Policy Enforcement",
    "Quality Assurance", "Testing Automation", "Validation", "Verification",
    "Process Optimization", "Efficiency Improvement", "Cost Reduction", "Time Savings",
    "Resource Optimization", "Capacity Planning", "Load Balancing", "Performance Tuning",
    "Integration Testing", "End-to-end Testing", "Regression Testing", "User Acceptance Testing",
    "Change Management", "Configuration Management", "Release Management", "Deployment Management",
    "Incident Management", "Problem Management", "Service Management", "IT Service Management",
    "Help Desk Integration", "Ticketing System", "Issue Tracking", "Resolution Management",
    "SLA Management", "Performance Metrics", "KPI Tracking", "Dashboard Creation",
    "Reporting", "Analytics", "Business Intelligence", "Data Visualization",
    "Custom Dashboards", "Real-time Monitoring", "Alerting", "Notification Systems",
    
    # === OPTIMIZATION & MATHEMATICAL MODELING ===
    "Mathematical Optimization", "Linear Programming", "Integer Programming", "Mixed Integer Programming",
    "Convex Optimization", "Non-linear Optimization", "Stochastic Optimization", "Robust Optimization",
    "Multi-objective Optimization", "Evolutionary Algorithms", "Genetic Algorithms", "Simulated Annealing",
    "Particle Swarm Optimization", "Ant Colony Optimization", "Differential Evolution", "Bayesian Optimization",
    "Hyperparameter Optimization", "Neural Architecture Search", "AutoML", "Feature Selection",
    "Dimensionality Reduction", "Principal Component Analysis", "Independent Component Analysis",
    "t-SNE", "UMAP", "Clustering", "K-means", "Hierarchical Clustering", "DBSCAN",
    "Gaussian Mixture Models", "Spectral Clustering", "Fuzzy Clustering", "Density-based Clustering",
    "Classification", "Regression", "Logistic Regression", "Decision Trees", "Random Forest",
    "Support Vector Machines", "Naive Bayes", "K-Nearest Neighbors", "Ensemble Methods",
    "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "AdaBoost", "Bagging",
    "Stacking", "Voting", "Blending", "Model Selection", "Cross-validation", "Grid Search",
    "Random Search", "Bayesian Search", "Model Evaluation", "Performance Metrics", "ROC Curves",
    "Precision-Recall Curves", "Confusion Matrix", "F1 Score", "Accuracy", "Precision",
    "Recall", "Sensitivity", "Specificity", "AUC", "Log Loss", "Mean Squared Error",
    "Root Mean Squared Error", "Mean Absolute Error", "R-squared", "Adjusted R-squared",
    
    # === AUTONOMOUS AGENTS & ROBOTICS ===
    "Autonomous Agents", "Multi-agent Systems", "Agent Communication", "Agent Coordination",
    "Agent Negotiation", "Agent Learning", "Reinforcement Learning", "Deep Reinforcement Learning",
    "Q-Learning", "Policy Gradient", "Actor-Critic", "TD Learning", "Monte Carlo Methods",
    "Markov Decision Processes", "Partially Observable MDPs", "Multi-agent Reinforcement Learning",
    "Cooperative Learning", "Competitive Learning", "Game Theory", "Nash Equilibrium",
    "Mechanism Design", "Auction Theory", "Social Choice", "Voting Systems",
    "Consensus Mechanisms", "Distributed Decision Making", "Swarm Intelligence", "Collective Intelligence",
    "Emergent Behavior", "Self-organization", "Adaptation", "Evolution", "Coevolution",
    "Artificial Life", "Cellular Automata", "Agent-based Modeling", "Complex Adaptive Systems",
    "Robotics", "Robot Control", "Robot Navigation", "Path Planning", "Motion Planning",
    "Trajectory Optimization", "Obstacle Avoidance", "Collision Detection", "Mapping", "SLAM",
    "Localization", "Sensor Fusion", "Computer Vision for Robotics", "Robot Perception",
    "Manipulation", "Grasping", "Dexterous Manipulation", "Force Control", "Compliance Control",
    "Humanoid Robotics", "Bipedal Walking", "Balance Control", "Gait Generation", "Locomotion",
    "Drone Control", "UAV Navigation", "Aerial Robotics", "Autonomous Vehicles", "Self-driving Cars",
    "Autonomous Navigation", "Traffic Management", "Route Planning", "Fleet Management",
    "Logistics Optimization", "Supply Chain Automation", "Warehouse Robotics", "Inventory Management",
    "Robotic Process Automation", "Industrial Automation", "Manufacturing Automation", "Quality Control",
    "Predictive Maintenance", "Condition Monitoring", "Asset Management", "Equipment Optimization",
    
    # === SCIENTIFIC & DOMAIN-SPECIFIC AI ===
    "Drug Discovery", "Molecular Design", "Protein Folding", "Protein Structure Prediction",
    "Genomics", "Proteomics", "Metabolomics", "Bioinformatics", "Computational Biology",
    "Systems Biology", "Synthetic Biology", "Gene Expression Analysis", "DNA Sequencing",
    "RNA Analysis", "Microarray Analysis", "Single-cell Analysis", "Phylogenetic Analysis",
    "Evolutionary Analysis", "Population Genetics", "Epidemiology", "Disease Modeling",
    "Clinical Trial Design", "Personalized Medicine", "Precision Medicine", "Biomarker Discovery",
    "Diagnostic Imaging", "Medical Imaging", "Radiology", "Pathology", "Histopathology",
    "Cytopathology", "Dermatology", "Ophthalmology", "Cardiology", "Oncology", "Neurology",
    "Psychiatry", "Mental Health", "Cognitive Assessment", "Behavioral Analysis", "Therapy",
    "Rehabilitation", "Physical Therapy", "Occupational Therapy", "Speech Therapy",
    "Telemedicine", "Remote Monitoring", "Wearable Technology", "Health Tracking", "Fitness Tracking",
    "Nutrition Analysis", "Diet Planning", "Meal Planning", "Recipe Generation", "Food Analysis",
    "Agriculture", "Crop Monitoring", "Yield Prediction", "Pest Detection", "Disease Detection",
    "Soil Analysis", "Weather Prediction", "Climate Modeling", "Environmental Monitoring",
    "Pollution Detection", "Air Quality Monitoring", "Water Quality Monitoring", "Ecosystem Modeling",
    "Biodiversity Analysis", "Species Identification", "Conservation", "Wildlife Monitoring",
    "Forestry", "Deforestation Monitoring", "Carbon Sequestration", "Renewable Energy",
    "Solar Energy", "Wind Energy", "Hydroelectric", "Energy Storage", "Smart Grid",
    "Energy Efficiency", "Building Automation", "HVAC Control", "Lighting Control", "Security Systems",
    "Access Control", "Surveillance", "Intrusion Detection", "Perimeter Security", "Facial Recognition",
    "License Plate Recognition", "Behavioral Analysis", "Crowd Control", "Emergency Response",
    "Disaster Management", "Risk Assessment", "Threat Analysis", "Vulnerability Assessment",
    "Cybersecurity", "Network Security", "Endpoint Security", "Application Security", "Data Security",
    "Privacy Protection", "Anonymization", "Differential Privacy", "Homomorphic Encryption",
    "Secure Multi-party Computation", "Zero-knowledge Proofs", "Blockchain", "Cryptocurrency",
    "Smart Contracts", "Decentralized Applications", "Distributed Ledger", "Consensus Algorithms",
    "Cryptographic Protocols", "Digital Signatures", "Hash Functions", "Random Number Generation",
    
    # === ADVANCED AI TECHNIQUES ===
    "Deep Learning", "Neural Networks", "Convolutional Neural Networks", "Recurrent Neural Networks",
    "Long Short-Term Memory", "Gated Recurrent Units", "Transformer Networks", "Attention Mechanisms",
    "Self-attention", "Multi-head Attention", "Positional Encoding", "Encoder-Decoder Architecture",
    "Sequence-to-Sequence", "Variational Autoencoders", "Generative Adversarial Networks",
    "Diffusion Models", "Flow-based Models", "Energy-based Models", "Normalizing Flows",
    "Neural Ordinary Differential Equations", "Graph Neural Networks", "Graph Convolution",
    "Graph Attention Networks", "Message Passing", "Node Classification", "Link Prediction",
    "Graph Generation", "Knowledge Graphs", "Ontologies", "Semantic Web", "Linked Data",
    "RDF", "SPARQL", "OWL", "Description Logic", "Reasoning", "Inference", "Deduction",
    "Induction", "Abduction", "Probabilistic Reasoning", "Bayesian Networks", "Markov Networks",
    "Belief Networks", "Causal Networks", "Causal Inference", "Causal Discovery", "Counterfactuals",
    "Structural Causal Models", "Do-calculus", "Instrumental Variables", "Confounding", "Bias",
    "Fairness", "Algorithmic Fairness", "Discrimination", "Equity", "Inclusion", "Diversity",
    "Bias Detection", "Bias Mitigation", "Fairness Metrics", "Demographic Parity", "Equalized Odds",
    "Calibration", "Individual Fairness", "Group Fairness", "Counterfactual Fairness",
    "Explainable AI", "Interpretable AI", "Model Interpretability", "Feature Importance",
    "SHAP", "LIME", "Permutation Importance", "Partial Dependence Plots", "ANCHORS",
    "Counterfactual Explanations", "Adversarial Examples", "Robustness", "Adversarial Training",
    "Defensive Distillation", "Adversarial Detection", "Certified Defenses", "Robustness Verification",
    "Uncertainty Quantification", "Epistemic Uncertainty", "Aleatoric Uncertainty", "Bayesian Deep Learning",
    "Monte Carlo Dropout", "Ensemble Uncertainty", "Conformal Prediction", "Calibration",
    "Out-of-distribution Detection", "Novelty Detection", "One-class Classification", "Isolation Forest",
    "Local Outlier Factor", "Autoencoder Anomaly Detection", "Statistical Anomaly Detection",
    
    # === BUSINESS & ENTERPRISE AI ===
    "Customer Relationship Management", "Sales Automation", "Lead Generation", "Lead Scoring",
    "Customer Segmentation", "Customer Lifetime Value", "Churn Prediction", "Retention Modeling",
    "Cross-selling", "Upselling", "Price Optimization", "Dynamic Pricing", "Revenue Management",
    "Inventory Optimization", "Demand Forecasting", "Supply Chain Optimization", "Procurement",
    "Vendor Management", "Contract Management", "Risk Management", "Compliance", "Audit",
    "Fraud Detection", "Anti-money Laundering", "Know Your Customer", "Identity Verification",
    "Credit Scoring", "Loan Approval", "Insurance Underwriting", "Claims Processing", "Actuarial Modeling",
    "Financial Planning", "Investment Analysis", "Portfolio Management", "Algorithmic Trading",
    "High-frequency Trading", "Market Making", "Risk Modeling", "Stress Testing", "Scenario Analysis",
    "Regulatory Reporting", "Basel III", "Solvency II", "GDPR Compliance", "Data Governance",
    "Master Data Management", "Data Quality", "Data Lineage", "Data Catalog", "Metadata Management",
    "Data Integration", "ETL", "ELT", "Data Warehousing", "Data Lakes", "Data Mesh",
    "Real-time Analytics", "Stream Processing", "Event Processing", "Complex Event Processing",
    "Time Series Analysis", "Anomaly Detection", "Pattern Recognition", "Trend Analysis",
    "Seasonality Analysis", "Forecasting", "Capacity Planning", "Resource Allocation", "Scheduling",
    "Workforce Management", "Talent Acquisition", "Candidate Screening", "Resume Parsing",
    "Interview Scheduling", "Performance Management", "Employee Engagement", "Retention Analysis",
    "Succession Planning", "Learning and Development", "Training Optimization", "Skill Gap Analysis",
    "Competency Management", "360-degree Feedback", "Peer Review", "Performance Appraisal",
    "Compensation Analysis", "Benefits Optimization", "Payroll Processing", "Time and Attendance",
    "Leave Management", "Compliance Tracking", "Safety Management", "Incident Reporting",
    "Workplace Analytics", "Productivity Analysis", "Collaboration Analysis", "Communication Analysis",
    "Sentiment Analysis", "Pulse Surveys", "Employee Feedback", "Culture Assessment", "Diversity Metrics",
    
    # === CREATIVE & ENTERTAINMENT AI ===
    "Creative Writing", "Story Generation", "Plot Development", "Character Creation", "Dialogue Generation",
    "Poetry Generation", "Songwriting", "Lyric Generation", "Screenplay Writing", "Script Generation",
    "Content Creation", "Blog Writing", "Article Writing", "Copywriting", "Marketing Copy",
    "Ad Copy", "Email Marketing", "Social Media Content", "SEO Content", "Technical Writing",
    "Documentation", "User Manuals", "API Documentation", "Tutorial Creation", "Course Content",
    "Educational Materials", "Lesson Planning", "Curriculum Development", "Assessment Creation",
    "Quiz Generation", "Test Creation", "Grading", "Feedback Generation", "Personalized Learning",
    "Adaptive Learning", "Learning Analytics", "Student Performance", "Engagement Analysis",
    "Dropout Prediction", "Intervention Strategies", "Tutoring Systems", "Intelligent Tutoring",
    "Virtual Tutors", "Chatbot Tutors", "Language Learning", "Translation", "Pronunciation",
    "Accent Training", "Conversation Practice", "Grammar Correction", "Vocabulary Building",
    "Reading Comprehension", "Listening Comprehension", "Speaking Practice", "Writing Practice",
    "Game Development", "Game Design", "Level Generation", "Procedural Generation", "Asset Generation",
    "Character Animation", "Facial Animation", "Motion Capture", "Gesture Animation", "Lip Sync",
    "Voice Acting", "Sound Design", "Music Composition", "Audio Production", "Sound Effects",
    "Ambient Sound", "Interactive Audio", "Adaptive Music", "Dynamic Soundtracks", "Audio Mixing",
    "Audio Mastering", "Podcast Production", "Radio Production", "Broadcasting", "Live Streaming",
    "Video Production", "Video Editing", "Color Grading", "Visual Effects", "Motion Graphics",
    "Animation", "2D Animation", "3D Animation", "Stop Motion", "Rotoscoping", "Compositing",
    "Rendering", "Lighting", "Shading", "Texturing", "Modeling", "Rigging", "Skinning",
    "Simulation", "Particle Systems", "Fluid Simulation", "Cloth Simulation", "Hair Simulation",
    "Crowd Simulation", "Physics Simulation", "Destruction", "Explosions", "Fire", "Smoke",
    "Virtual Reality", "Augmented Reality", "Mixed Reality", "360-degree Video", "Immersive Experiences",
    "Interactive Media", "Transmedia", "Cross-platform", "Multi-device", "Responsive Design",
    "User Interface Design", "User Experience Design", "Interaction Design", "Visual Design",
    "Graphic Design", "Logo Design", "Brand Design", "Website Design", "App Design",
    "Dashboard Design", "Data Visualization", "Infographic Design", "Presentation Design",
    "Print Design", "Packaging Design", "Product Design", "Industrial Design", "Fashion Design",
    "Interior Design", "Architecture", "Urban Planning", "Landscape Architecture", "Environmental Design"
]

def advanced_feature_extraction(soup, url, visited_urls=None):
    """Enhanced feature extraction with better pattern recognition"""
    global MASTER_FEATURES
    if visited_urls is None:
        visited_urls = set()
    
    features = []
    
    # Advanced keyword mapping with context
    feature_patterns = {
        'video': ['Scene Detection', 'Auto-trimming', 'Video Stabilization', 'Color Grading', 
                 'Frame Interpolation', 'Motion Tracking', 'Video Compression', 'Subtitle Generation'],
        'audio': ['Audio Enhancement', 'Noise Reduction', 'Speech Diarization', 'Audio Mixing',
                 'Sound Design', 'Music Composition', 'Voice Cloning', 'Audio Restoration'],
        'text': ['Text Generation', 'Sentiment Analysis', 'Named Entity Recognition', 'Grammar Correction',
                'Text Summarization', 'Language Translation', 'Keyword Extraction', 'Content Optimization'],
        'image': ['Object Detection', 'Facial Recognition', 'Image Enhancement', 'Style Transfer',
                 'Background Removal', 'Image Segmentation', 'OCR', 'Visual Search'],
        'ai': ['Machine Learning', 'Deep Learning', 'Neural Networks', 'Natural Language Processing',
              'Computer Vision', 'Predictive Analytics', 'Recommendation Systems', 'Chatbots'],
        'business': ['CRM Integration', 'Sales Automation', 'Lead Generation', 'Customer Analytics',
                    'Revenue Optimization', 'Inventory Management', 'Supply Chain', 'Risk Management'],
        'security': ['Fraud Detection', 'Cybersecurity', 'Threat Detection', 'Compliance Monitoring',
                    'Identity Verification', 'Access Control', 'Data Protection', 'Privacy'],
        'automation': ['Workflow Automation', 'Process Optimization', 'Task Automation', 'RPA',
                      'Scheduling', 'Resource Allocation', 'Performance Monitoring', 'Quality Control']
    }
    
    # Extract comprehensive text content
    text_content = []
    for tag in ['title', 'meta', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'span', 'a']:
        elements = soup.find_all(tag)
        for element in elements:
            if tag == 'meta':
                content = element.get('content', '') or element.get('name', '')
            else:
                content = element.get_text(strip=True)
            if content:
                text_content.append(content.lower())
    
    # Pattern matching for feature extraction
    full_text = ' '.join(text_content)
    
    for keyword, related_features in feature_patterns.items():
        if keyword in full_text:
            for feature in related_features:
                if feature not in features:
                    features.append(feature)
    
    # Advanced semantic analysis
    for feature in MASTER_FEATURES:
        if feature.lower() in full_text:
            if feature not in features:
                features.append(feature)
    
    # Extract from structured data (JSON-LD, microdata)
    json_ld_scripts = soup.find_all('script', type='application/ld+json')
    for script in json_ld_scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                description = data.get('description', '')
                if description:
                    for keyword, related_features in feature_patterns.items():
                        if keyword in description.lower():
                            features.extend(related_features)
        except:
            pass
    
    return list(set(features))

def enhanced_service_analysis(soup, url):
    """Enhanced service analysis with better accuracy"""
    analysis = {}
    
    # Extract all text content
    text_content = soup.get_text().lower()
    
    # Advanced pricing detection
    pricing_patterns = {
        'free': r'\b(free|no cost|zero cost|complimentary)\b',
        'freemium': r'\b(freemium|free plan|free tier|free version)\b',
        'subscription': r'\b(subscription|monthly|annually|recurring)\b',
        'pay_per_use': r'\b(pay per use|pay as you go|usage based|per request)\b',
        'one_time': r'\b(one time|lifetime|perpetual license)\b',
        'enterprise': r'\b(enterprise|custom pricing|contact sales)\b'
    }
    
    pricing_model = "Unknown"
    for model, pattern in pricing_patterns.items():
        if re.search(pattern, text_content):
            pricing_model = model.replace('_', ' ').title()
            break
    
    # Extract pricing amounts
    price_matches = re.findall(r'\$\d+(?:\.\d{2})?(?:/\w+)?', text_content)
    pricing_details = ', '.join(price_matches[:3]) if price_matches else "No specific pricing found"
    
    # Performance metrics detection
    performance_patterns = {
        'speed': r'(\d+(?:\.\d+)?)\s*(ms|seconds?|minutes?|hours?)',
        'accuracy': r'(\d+(?:\.\d+)?)\s*%\s*accuracy',
        'uptime': r'(\d+(?:\.\d+)?)\s*%\s*uptime',
        'throughput': r'(\d+(?:,\d{3})*)\s*(requests?|transactions?|operations?)\s*per\s*(second|minute|hour)'
    }
    
    performance_metrics = {}
    for metric, pattern in performance_patterns.items():
        matches = re.findall(pattern, text_content)
        if matches:
            performance_metrics[metric] = matches[0]
    
    # API and integration detection
    api_features = []
    if 'api' in text_content:
        api_features.append('REST API')
    if 'webhook' in text_content:
        api_features.append('Webhooks')
    if 'sdk' in text_content:
        api_features.append('SDK')
    if 'integration' in text_content:
        api_features.append('Third-party Integrations')
    
    analysis.update({
        'pricing_model': pricing_model,
        'pricing_details': pricing_details,
        'performance_metrics': performance_metrics,
        'api_features': api_features
    })
    
    return analysis

def extract_service_info_enhanced(url):
    """Enhanced service information extraction"""
    if not can_scrape(url):
        print(f"Scraping not allowed for {url} per robots.txt")
        return None
    
    content = get_page_content(url)
    if not content:
        return None
    
    soup = BeautifulSoup(content, "html.parser")
    
    # Enhanced service name extraction
    service_name = "Unknown"
    title_tag = soup.find("title")
    if title_tag:
        service_name = title_tag.get_text(strip=True)
    
    # Try to get brand name from various sources
    brand_selectors = [
        'meta[property="og:site_name"]',
        'meta[name="application-name"]',
        '.brand',
        '.logo',
        'h1'
    ]
    
    for selector in brand_selectors:
        element = soup.select_one(selector)
        if element:
            brand_text = element.get('content') or element.get_text(strip=True)
            if brand_text and len(brand_text) < 100:
                service_name = brand_text
                break
    
    # Enhanced provider extraction
    provider = "Unknown"
    provider_selectors = [
        'meta[name="author"]',
        'meta[property="og:site_name"]',
        'meta[name="company"]',
        '.company',
        '.author'
    ]
    
    for selector in provider_selectors:
        element = soup.select_one(selector)
        if element:
            provider_text = element.get('content') or element.get_text(strip=True)
            if provider_text:
                provider = provider_text
                break
    
    # Enhanced description extraction
    description = "No description available."
    description_selectors = [
        'meta[name="description"]',
        'meta[property="og:description"]',
        'meta[name="twitter:description"]',
        '.description',
        '.summary'
    ]
    
    for selector in description_selectors:
        element = soup.select_one(selector)
        if element:
            desc_text = element.get('content') or element.get_text(strip=True)
            if desc_text:
                description = desc_text[:300] + "..." if len(desc_text) > 300 else desc_text
                break
    
    # Enhanced feature extraction
    features = advanced_feature_extraction(soup, url)
    
    # Enhanced service analysis
    service_analysis = enhanced_service_analysis(soup, url)
    
    # Category and type determination
    category_group, category_name, category_type = determine_category_and_type(description, features)
    service_type = determine_service_type(features)
    
    # Platform compatibility detection
    platform_compatibility = ["Web App"]  # Default
    
    platform_indicators = {
        'api': 'API',
        'chrome extension': 'Chrome Extension',
        'firefox extension': 'Firefox Extension',
        'mobile app': 'Mobile App',
        'ios app': 'iOS App',
        'android app': 'Android App',
        'desktop app': 'Desktop App',
        'windows': 'Windows App',
        'mac': 'Mac App',
        'linux': 'Linux App'
    }
    
    text_content = soup.get_text().lower()
    for indicator, platform in platform_indicators.items():
        if indicator in text_content and platform not in platform_compatibility:
            platform_compatibility.append(platform)
    
    # Build the comprehensive service info
    service_info = {
        "Service_Name": service_name,
        "Service_URL": url,
        "Provider": provider,
        "Description": description,
        "Category_Group": category_group,
        "Category_Name": category_name,
        "Category_Type": category_type,
        "Service_Type": service_type,
        "Features": sorted(list(set(features))),
        "Pricing_Model": service_analysis.get('pricing_model', 'Unknown'),
        "Pricing_Details": service_analysis.get('pricing_details', 'No pricing information found'),
        "Performance_Metrics": service_analysis.get('performance_metrics', {}),
        "API_Features": service_analysis.get('api_features', []),
        "Platform_Compatibility": platform_compatibility,
        "Free_Trial": "free trial" in text_content or "trial" in text_content,
        "Underlying_Model": extract_model_info(soup),
        "Carbon_Footprint": extract_sustainability_info(soup),
        "Source_File": url,
        "Extraction_Date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return service_info

def extract_model_info(soup):
    """Extract information about underlying AI models"""
    text_content = soup.get_text().lower()
    
    model_patterns = {
        'gpt': r'\b(gpt-?[0-9\.]+|gpt|generative pre-trained transformer)\b',
        'bert': r'\b(bert|bidirectional encoder representations)\b',
        'llama': r'\b(llama|large language model meta ai)\b',
        'claude': r'\b(claude|anthropic)\b',
        'palm': r'\b(palm|pathways language model)\b',
        'proprietary': r'\b(proprietary|custom|in-house|internal)\b',
        'open_source': r'\b(open source|open-source|opensource)\b'
    }
    
    for model_type, pattern in model_patterns.items():
        if re.search(pattern, text_content):
            return model_type.replace('_', ' ').title()
    
    return "Non spécifié"

def extract_sustainability_info(soup):
    """Extract sustainability and environmental impact information"""
    text_content = soup.get_text().lower()
    
    sustainability_info = {
        "Carbon_Footprint": "Non spécifié",
        "Green_Computing": "Non spécifié",
        "Energy_Efficiency": "Non spécifié"
    }
    
    sustainability_patterns = {
        'carbon_neutral': r'\b(carbon neutral|net zero|carbon offset)\b',
        'renewable_energy': r'\b(renewable energy|solar|wind|green energy)\b',
        'energy_efficient': r'\b(energy efficient|low power|optimized)\b'
    }
    
    for key, pattern in sustainability_patterns.items():
        if re.search(pattern, text_content):
            sustainability_info[key.replace('_', ' ').title()] = "Mentioned"
    
    return sustainability_info

def generate_comprehensive_report(services):
    """Generate a comprehensive analysis report"""
    report = {
        "total_services": len(services),
        "categories": {},
        "service_types": {},
        "pricing_models": {},
        "top_features": {},
        "platform_distribution": {},
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Analyze categories
    for service in services:
        category = service.get('Category_Group', 'Unknown')
        report['categories'][category] = report['categories'].get(category, 0) + 1
    
    # Analyze service types
    for service in services:
        service_type = service.get('Service_Type', 'Unknown')
        report['service_types'][service_type] = report['service_types'].get(service_type, 0) + 1
    
    # Analyze pricing models
    for service in services:
        pricing = service.get('Pricing_Model', 'Unknown')
        report['pricing_models'][pricing] = report['pricing_models'].get(pricing, 0) + 1
    
    # Analyze top features
    feature_count = {}
    for service in services:
        for feature in service.get('Features', []):
            feature_count[feature] = feature_count.get(feature, 0) + 1
    
    report['top_features'] = dict(sorted(feature_count.items(), key=lambda x: x[1], reverse=True)[:20])
    
    # Analyze platform distribution
    platform_count = {}
    for service in services:
        for platform in service.get('Platform_Compatibility', []):
            platform_count[platform] = platform_count.get(platform, 0) + 1
    
    report['platform_distribution'] = platform_count
    
    return report

def main():
    """Enhanced main function with comprehensive analysis"""
    input_file = "urls.txt"
    output_file = "ai_services_comprehensive.json"
    report_file = "ai_services_report.json"
    services = []
    
    try:
        with open(input_file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return
    
    total_urls = len(urls)
    print(f"Processing {total_urls} URLs...")
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{total_urls}] Processing {url}...")
        service_info = extract_service_info_enhanced(url)
        if service_info:
            services.append(service_info)
            print(f"  ✓ Extracted {len(service_info['Features'])} features")
        else:
            print(f"  ✗ Failed to extract information")
        
        time.sleep(1)  # Respectful delay
    
    # Generate comprehensive report
    report = generate_comprehensive_report(services)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(services, f, indent=4, ensure_ascii=False)
    print(f"\nExtracted {len(services)} services to {output_file}")
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"Generated comprehensive report to {report_file}")

if __name__ == "__main__":
    main()