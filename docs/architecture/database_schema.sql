-- Enterprise AI Document Intelligence Platform Database Schema
-- Designed for multi-tenant SaaS architecture with audit trails

-- Document Management Tables
CREATE TABLE organizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    api_key TEXT UNIQUE NOT NULL,
    tier TEXT DEFAULT 'basic',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL, -- 'invoice', 'purchase_order', 'contract', etc.
    description TEXT,
    extraction_rules JSON,
    validation_schema JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id INTEGER NOT NULL,
    document_type_id INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    mime_type TEXT,
    upload_source TEXT, -- 'api', 'webhook', 'batch_upload'
    status TEXT DEFAULT 'processing', -- 'processing', 'completed', 'failed', 'needs_review'
    confidence_score REAL,
    processing_started_at DATETIME,
    processing_completed_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations (id),
    FOREIGN KEY (document_type_id) REFERENCES document_types (id)
);

-- Processing Results and Extracted Data
CREATE TABLE extraction_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    extraction_method TEXT, -- 'regex', 'claude', 'azure_ocr', etc.
    raw_data JSON NOT NULL,
    structured_data JSON NOT NULL,
    confidence_score REAL,
    processing_time_ms INTEGER,
    validation_errors JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id)
);

-- Agent Performance and Learning
CREATE TABLE agent_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    session_type TEXT, -- 'individual', 'collaborative', 'competitive'
    document_id INTEGER,
    task_description TEXT,
    strategy_used TEXT,
    execution_time_ms INTEGER,
    success BOOLEAN,
    confidence_score REAL,
    tokens_used INTEGER DEFAULT 0,
    cost_usd DECIMAL(10,4) DEFAULT 0.0000,
    memory_usage_mb INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id)
);

-- Multi-Agent Coordination Tracking
CREATE TABLE orchestration_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    orchestrator_name TEXT NOT NULL,
    coordination_pattern TEXT, -- 'hierarchical', 'parallel', 'consensus', 'swarm'
    participating_agents JSON,
    task_description TEXT,
    total_execution_time_ms INTEGER,
    final_result JSON,
    consensus_achieved BOOLEAN,
    emergent_behaviors JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Business Intelligence and Analytics
CREATE TABLE processing_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id INTEGER NOT NULL,
    document_type_id INTEGER,
    processing_date DATE NOT NULL,
    documents_processed INTEGER DEFAULT 0,
    successful_extractions INTEGER DEFAULT 0,
    average_confidence REAL,
    average_processing_time_ms INTEGER,
    total_cost_usd DECIMAL(10,4) DEFAULT 0.0000,
    anomalies_detected INTEGER DEFAULT 0,
    human_reviews_required INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations (id),
    FOREIGN KEY (document_type_id) REFERENCES document_types (id)
);

-- Human-in-the-Loop Workflow
CREATE TABLE review_queues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    assigned_reviewer_id TEXT,
    priority INTEGER DEFAULT 1, -- 1=low, 2=medium, 3=high, 4=critical
    reason_for_review TEXT, -- 'low_confidence', 'anomaly_detected', 'compliance_check'
    status TEXT DEFAULT 'pending', -- 'pending', 'in_review', 'approved', 'rejected'
    reviewer_feedback JSON,
    reviewed_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id)
);

-- Integration and Webhook Management
CREATE TABLE integrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id INTEGER NOT NULL,
    integration_type TEXT NOT NULL, -- 'quickbooks', 'sap', 'netsuite', 'webhook'
    configuration JSON NOT NULL,
    status TEXT DEFAULT 'active', -- 'active', 'inactive', 'error'
    last_sync_at DATETIME,
    error_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations (id)
);

CREATE TABLE webhook_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id INTEGER NOT NULL,
    event_type TEXT NOT NULL, -- 'document.processed', 'review.completed', 'anomaly.detected'
    payload JSON NOT NULL,
    webhook_url TEXT NOT NULL,
    status TEXT DEFAULT 'pending', -- 'pending', 'sent', 'failed'
    attempts INTEGER DEFAULT 0,
    last_attempt_at DATETIME,
    response_code INTEGER,
    response_body TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations (id)
);

-- Compliance and Audit Trail
CREATE TABLE audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id INTEGER NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL, -- 'document.uploaded', 'data.extracted', 'review.completed'
    resource_type TEXT, -- 'document', 'agent', 'integration'
    resource_id INTEGER,
    old_values JSON,
    new_values JSON,
    ip_address TEXT,
    user_agent TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations (id)
);

-- Performance Optimization Tables
CREATE TABLE extraction_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_type TEXT NOT NULL,
    field_name TEXT NOT NULL,
    pattern_type TEXT, -- 'regex', 'ml_model', 'rule_based'
    pattern_data JSON NOT NULL,
    accuracy_score REAL,
    usage_count INTEGER DEFAULT 0,
    last_used_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance
CREATE INDEX idx_documents_org_status ON documents(organization_id, status);
CREATE INDEX idx_documents_type_date ON documents(document_type_id, created_at);
CREATE INDEX idx_extraction_results_doc ON extraction_results(document_id);
CREATE INDEX idx_agent_sessions_agent_date ON agent_sessions(agent_name, created_at);
CREATE INDEX idx_analytics_org_date ON processing_analytics(organization_id, processing_date);
CREATE INDEX idx_review_queues_status ON review_queues(status, priority);
CREATE INDEX idx_webhook_events_status ON webhook_events(status, created_at);
CREATE INDEX idx_audit_logs_org_date ON audit_logs(organization_id, created_at);

-- Views for Common Queries
CREATE VIEW v_organization_performance AS
SELECT 
    o.name as organization_name,
    COUNT(d.id) as total_documents,
    AVG(er.confidence_score) as avg_confidence,
    SUM(CASE WHEN d.status = 'completed' THEN 1 ELSE 0 END) as successful_docs,
    COUNT(rq.id) as pending_reviews
FROM organizations o
LEFT JOIN documents d ON o.id = d.organization_id
LEFT JOIN extraction_results er ON d.id = er.document_id
LEFT JOIN review_queues rq ON d.id = rq.document_id AND rq.status = 'pending'
GROUP BY o.id, o.name;

CREATE VIEW v_agent_efficiency AS
SELECT 
    agent_name,
    COUNT(*) as total_sessions,
    AVG(confidence_score) as avg_confidence,
    AVG(execution_time_ms) as avg_processing_time,
    SUM(tokens_used) as total_tokens,
    SUM(cost_usd) as total_cost
FROM agent_sessions
WHERE created_at >= date('now', '-30 days')
GROUP BY agent_name;