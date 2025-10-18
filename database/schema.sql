-- Database schema for Ad Headline Optimizer
-- Supports both PostgreSQL and SQLite

-- Campaigns table
CREATE TABLE IF NOT EXISTS campaigns (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_audience VARCHAR(255),
    placement VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Headlines table
CREATE TABLE IF NOT EXISTS headlines (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    headline_text TEXT NOT NULL,
    variant_type VARCHAR(50) DEFAULT 'generated', -- 'original', 'generated', 'optimized'
    tone VARCHAR(50),
    placement VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Features table (for storing extracted features)
CREATE TABLE IF NOT EXISTS headline_features (
    id SERIAL PRIMARY KEY,
    headline_id INTEGER REFERENCES headlines(id),
    char_count INTEGER,
    word_count INTEGER,
    sentiment_polarity FLOAT,
    sentiment_subjectivity FLOAT,
    readability_score FLOAT,
    power_word_count INTEGER,
    question_count INTEGER,
    exclamation_count INTEGER,
    uppercase_ratio FLOAT,
    digit_count INTEGER,
    special_char_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- CTR predictions table
CREATE TABLE IF NOT EXISTS ctr_predictions (
    id SERIAL PRIMARY KEY,
    headline_id INTEGER REFERENCES headlines(id),
    predicted_ctr FLOAT NOT NULL,
    model_version VARCHAR(50),
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- A/B tests table
CREATE TABLE IF NOT EXISTS ab_tests (
    id SERIAL PRIMARY KEY,
    test_id VARCHAR(100) UNIQUE NOT NULL,
    campaign_id INTEGER REFERENCES campaigns(id),
    variant_a_id INTEGER REFERENCES headlines(id),
    variant_b_id INTEGER REFERENCES headlines(id),
    target_impressions INTEGER,
    significance_level FLOAT DEFAULT 0.05,
    power FLOAT DEFAULT 0.8,
    min_sample_size INTEGER DEFAULT 1000,
    max_duration_days INTEGER DEFAULT 30,
    early_stopping BOOLEAN DEFAULT TRUE,
    status VARCHAR(50) DEFAULT 'created', -- 'created', 'running', 'completed', 'stopped'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- A/B test results table
CREATE TABLE IF NOT EXISTS ab_test_results (
    id SERIAL PRIMARY KEY,
    test_id VARCHAR(100) REFERENCES ab_tests(test_id),
    impressions_a INTEGER,
    impressions_b INTEGER,
    clicks_a INTEGER,
    clicks_b INTEGER,
    ctr_a FLOAT,
    ctr_b FLOAT,
    lift FLOAT,
    p_value FLOAT,
    is_significant BOOLEAN,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    test_duration_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    dataset_type VARCHAR(50), -- 'train', 'validation', 'test'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature importance table
CREATE TABLE IF NOT EXISTS feature_importance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    importance_score FLOAT NOT NULL,
    rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User interactions table (for tracking usage)
CREATE TABLE IF NOT EXISTS user_interactions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    action_type VARCHAR(100), -- 'generate', 'predict', 'test', 'view'
    headline_id INTEGER REFERENCES headlines(id),
    test_id VARCHAR(100) REFERENCES ab_tests(test_id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cost tracking table
CREATE TABLE IF NOT EXISTS cost_tracking (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    service VARCHAR(50), -- 'openai', 'anthropic', 'redis', 'database'
    operation VARCHAR(100),
    tokens_used INTEGER,
    cost_usd FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_headlines_campaign_id ON headlines(campaign_id);
CREATE INDEX IF NOT EXISTS idx_headlines_created_at ON headlines(created_at);
CREATE INDEX IF NOT EXISTS idx_headline_features_headline_id ON headline_features(headline_id);
CREATE INDEX IF NOT EXISTS idx_ctr_predictions_headline_id ON ctr_predictions(headline_id);
CREATE INDEX IF NOT EXISTS idx_ctr_predictions_created_at ON ctr_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_ab_tests_campaign_id ON ab_tests(campaign_id);
CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON ab_tests(status);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_test_id ON ab_test_results(test_id);
CREATE INDEX IF NOT EXISTS idx_model_performance_model_version ON model_performance(model_version);
CREATE INDEX IF NOT EXISTS idx_feature_importance_model_version ON feature_importance(model_version);
CREATE INDEX IF NOT EXISTS idx_user_interactions_session_id ON user_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_created_at ON user_interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_session_id ON cost_tracking(session_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_created_at ON cost_tracking(created_at);

-- Views for common queries

-- Headlines with features and predictions
CREATE OR REPLACE VIEW headlines_with_features AS
SELECT 
    h.id,
    h.campaign_id,
    h.headline_text,
    h.variant_type,
    h.tone,
    h.placement,
    h.created_at,
    hf.char_count,
    hf.word_count,
    hf.sentiment_polarity,
    hf.sentiment_subjectivity,
    hf.readability_score,
    hf.power_word_count,
    hf.question_count,
    hf.exclamation_count,
    hf.uppercase_ratio,
    hf.digit_count,
    hf.special_char_count,
    cp.predicted_ctr,
    cp.model_version,
    cp.confidence_score
FROM headlines h
LEFT JOIN headline_features hf ON h.id = hf.headline_id
LEFT JOIN ctr_predictions cp ON h.id = cp.headline_id;

-- A/B test summary
CREATE OR REPLACE VIEW ab_test_summary AS
SELECT 
    t.test_id,
    t.campaign_id,
    t.status,
    t.created_at,
    t.started_at,
    t.completed_at,
    ha.headline_text as variant_a,
    hb.headline_text as variant_b,
    r.impressions_a,
    r.impressions_b,
    r.clicks_a,
    r.clicks_b,
    r.ctr_a,
    r.ctr_b,
    r.lift,
    r.p_value,
    r.is_significant,
    r.confidence_interval_lower,
    r.confidence_interval_upper,
    r.test_duration_days
FROM ab_tests t
LEFT JOIN headlines ha ON t.variant_a_id = ha.id
LEFT JOIN headlines hb ON t.variant_b_id = hb.id
LEFT JOIN ab_test_results r ON t.test_id = r.test_id;

-- Campaign performance summary
CREATE OR REPLACE VIEW campaign_performance AS
SELECT 
    c.id as campaign_id,
    c.name as campaign_name,
    COUNT(DISTINCT h.id) as total_headlines,
    COUNT(DISTINCT t.id) as total_tests,
    AVG(cp.predicted_ctr) as avg_predicted_ctr,
    MAX(cp.predicted_ctr) as max_predicted_ctr,
    MIN(cp.predicted_ctr) as min_predicted_ctr,
    COUNT(CASE WHEN r.is_significant = true THEN 1 END) as significant_tests,
    AVG(CASE WHEN r.is_significant = true THEN r.lift END) as avg_significant_lift
FROM campaigns c
LEFT JOIN headlines h ON c.id = h.campaign_id
LEFT JOIN ctr_predictions cp ON h.id = cp.headline_id
LEFT JOIN ab_tests t ON c.id = t.campaign_id
LEFT JOIN ab_test_results r ON t.test_id = r.test_id
GROUP BY c.id, c.name;

-- Triggers for updating timestamps

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_headlines_updated_at BEFORE UPDATE ON headlines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- SQLite specific adjustments (for development)
-- Note: SQLite doesn't support SERIAL, so we use INTEGER PRIMARY KEY AUTOINCREMENT
-- Also, some PostgreSQL-specific features are not available in SQLite

-- For SQLite compatibility, you might want to create a separate schema file
-- or use conditional SQL based on the database type

-- Sample data for testing
INSERT OR IGNORE INTO campaigns (id, name, description, target_audience, placement) VALUES
(1, 'E-commerce Sale', 'Summer sale campaign for online store', 'Online shoppers', 'Facebook'),
(2, 'SaaS Trial', 'Free trial promotion for software product', 'Small business owners', 'Google Ads'),
(3, 'Mobile App', 'App download campaign', 'Mobile users', 'Instagram');

INSERT OR IGNORE INTO headlines (id, campaign_id, headline_text, variant_type, tone, placement) VALUES
(1, 1, 'Get 50% Off Today Only!', 'original', 'urgent', 'Facebook'),
(2, 1, 'Transform Your Life in 30 Days', 'generated', 'inspirational', 'Facebook'),
(3, 1, 'Free Shipping on All Orders', 'generated', 'informative', 'Facebook'),
(4, 2, 'Start Your Free Trial Today', 'original', 'action-oriented', 'Google Ads'),
(5, 2, 'Join Thousands of Happy Customers', 'generated', 'social proof', 'Google Ads'),
(6, 3, 'Download Our Amazing App', 'original', 'enthusiastic', 'Instagram'),
(7, 3, 'Get Started in Seconds', 'generated', 'simple', 'Instagram');
