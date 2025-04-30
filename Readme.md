# Resync - AI-Powered Research Collaboration Platform

A modern platform that revolutionizes research collaboration by integrating AI-driven paper analysis, real-time collaboration, and intelligent recommendations.

## Team Members - Invictus Hackathon
- Ayush Bohra
- Kunal Patil
- Vinit Solanki
- Prajwal Sawant

## Problem Statement
The existing research landscape is fragmented across multiple platforms, making it difficult for researchers to discover relevant work, access reliable datasets, and collaborate effectively. Our solution addresses:
- Fragmentation of Research Resources
- Limited Collaboration & Networking Opportunities
- Inefficient Discovery and Recommendation of Research

## Key Features

- 🤖 AI-Powered Paper Analysis
- 📚 Centralized Research Repository
- 🔍 Intelligent Search & Recommendations
- 👥 Real-time Collaboration Tools
- 💬 Interactive Chat with Papers
- 📊 Research Visualization
- 🔗 Cross-disciplinary Networking
- 📱 Responsive Design

## Quick Start

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
Create `.env` in backend directory:
```
MONGODB_URI=your_mongodb_uri
JWT_SECRET=your_jwt_secret
PORT=5000
```

3. Start development servers:

Backend:
```bash
cd backend
npm run dev
```

Frontend:
```bash
cd frontend
npm run dev
```

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```bash
# AWS Configuration
AWS_REGION=           # AWS region (e.g., ap-south-1)
AWS_ACCESS_KEY_ID=    # Your AWS access key
AWS_SECRET_ACCESS_KEY=# Your AWS secret key

# MongoDB Configuration
MONGODB_URI=         # Your MongoDB connection string

# JWT Configuration
JWT_SECRET=         # Secret key for JWT token generation
JWT_EXPIRES_IN=     # Token expiration time (e.g., 3d)

# Server Configuration
PORT=               # Server port number (e.g., 3000)
```

### Required Configurations:

1. **AWS Credentials**: Required for:
   - Document storage
   - File processing
   - Cloud services integration

2. **MongoDB**: 
   - Main database connection
   - Stores user data, research papers, and collaboration info

3. **JWT Settings**:
   - Handles user authentication
   - Manages session security

4. **Server Settings**:
   - Controls API server configuration
   - Manages application ports

> ⚠️ Note: Never commit the actual `.env` file to version control. Keep your credentials secure!

## project directory structure

```
.
├── Frontend (React + Vite)
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   │   ├── research/
│   │   │   ├── collaboration/
│   │   │   └── chat/
│   │   ├── context/
│   │   ├── hooks/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── types/
│   │   └── utils/
│   └── vite.config.ts
│
├── Backend (Node.js + Express)
│   ├── controllers/
│   │   ├── paperController.js
│   │   ├── collaborationController.js
│   │   └── userController.js
│   ├── models/
│   │   ├── Paper.js
│   │   ├── Collaboration.js
│   │   └── User.js
│   └── server.js
│
└── Model Server (Python + Flask)
    ├── server.py
    ├── uploads/
    │   └── papers/
    ├── utils/
    │   ├── pdf_processor.py
    │   └── ml_helpers.py
    └── requirements.txt
```

## System Architecture
![System Architecture](/image.jpg)

## Technologies Used


### Frontend
```
- React with TypeScript
- TailwindCSS for styling
- Socket.io for real-time features
- PDF.js for document viewing
```
### Backend
```
- Node.js & Express
- MongoDB for data storage
- WebSocket for real-time updates
- JWT authentication
```
### Model Server
```
- Flask for API endpoints
- Groq LLM for paper analysis
- PyPDF2 for PDF processing
- langchain for context management
```
## Key Solutions
1. **Centralized Repository**
   ```
   - Single platform for research papers
   - Integrated dataset management
   - Expert network connectivity
   ```
3. **AI-Driven Features**
   ```
   - Smart paper recommendations
   - Content summarization
   - Research trend analysis
   - Intelligent paper chat
   ```

5. **Collaboration Tools**
   ```
   - Real-time document annotation
   - Shared workspaces
   - Discussion forums
   - Expert networking
   ```

7. **Research Discovery**
   ```
   - Advanced semantic search
   - Cross-disciplinary recommendations
   - Citation network analysis
   - Research impact metrics
   ```

## Security & Scalability
```
- End-to-end encryption for sensitive data
- Microservices architecture for scalability
- Efficient data indexing and retrieval
- Regular security audits
```
