# Getting Started with Ad Headline Optimizer

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt
```

### Step 2: Set Up Environment
```bash
# Copy environment template
cp env.example .env

# Edit .env file with your settings (optional for basic demo)
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### Step 3: Run Basic Demo
```bash
# Test basic functionality
python test_simple.py

# Run comprehensive demo
python scripts/demo.py
```

### Step 4: Start Dashboard
```bash
# Launch Streamlit interface
streamlit run app/streamlit_app.py
```

## 🧪 Testing the System

### Test Individual Components

```bash
# Test Phase 1: Core Generation
python test_simple.py

# Test Phase 2: ML Pipeline
python test_simple_phase2.py

# Test Phase 3: A/B Testing
python test_phase3.py

# Test Phase 4: Production Features
python test_phase4.py
```

### Run Full System Demo

```bash
# Comprehensive demonstration
python scripts/comprehensive_demo.py
```

## 🎯 Basic Usage Examples

### 1. Generate Headlines (Without API Keys)

```python
from api.features import FeatureExtractor
from utils.diversity_filter import DiversityFilter

# Test headlines
headlines = [
    "Get 50% Off Today Only!",
    "Transform Your Life in 30 Days",
    "Free Shipping on All Orders"
]

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_features_batch(headlines)
print(f"Extracted {features.shape[1]} features")

# Filter for diversity
filter_obj = DiversityFilter()
diverse = filter_obj.ensure_diversity(headlines)
print(f"Filtered to {len(diverse)} diverse headlines")
```

### 2. Run A/B Test

```python
from experiments.ab_harness import ABTestHarness

# Create A/B test
harness = ABTestHarness()
config = harness.create_test(
    test_id="demo_test",
    variant_a="Get 50% Off Today Only!",
    variant_b="Transform Your Life in 30 Days",
    target_impressions=10000
)

# Simulate test results
result = harness.simulate_test(
    test_id="demo_test",
    true_ctr_a=0.05,
    true_ctr_b=0.06
)

print(f"Lift: {result.lift:.1%}, P-value: {result.p_value:.4f}")
```

### 3. Content Safety Check

```python
from utils.guardrails import ContentGuardrails

# Test content safety
guardrails = ContentGuardrails()
headlines = [
    "Get 50% Off Today Only!",
    "This is a fucking amazing deal!",  # Will be flagged
    "CLICK HERE NOW!!! FREE MONEY!!!"   # Will be flagged
]

results = guardrails.batch_filter(headlines)
safe_headlines = guardrails.get_safe_content(headlines)
print(f"Safe headlines: {safe_headlines}")
```

## 🏗 Advanced Setup

### Train the CTR Model

```bash
# Train with default settings
python models/train_ctr_model.py

# This will:
# 1. Generate 10,000 synthetic training samples
# 2. Extract features from headlines
# 3. Train LightGBM model with hyperparameter optimization
# 4. Save model to models/ctr_model.pkl
# 5. Generate performance plots
```

### Set Up Database (Optional)

```bash
# Install PostgreSQL
# Create database
createdb headline_optimizer

# Run schema
psql headline_optimizer < database/schema.sql
```

### Set Up Redis (Optional)

```bash
# Install Redis
# Start Redis server
redis-server

# Test connection
redis-cli ping
```

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build and run all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

### Access Services

- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **Database**: localhost:5432
- **Redis**: localhost:6379

## 📊 Understanding the Output

### Feature Extraction Results
```
✅ Extracted 25 features for 5 headlines
📊 Feature columns: ['char_count', 'word_count', 'sentiment_polarity', ...]
📈 Sample features:
   char_count  word_count  sentiment_polarity  readability_score
0          20           4               0.200              85.2
1          28           6               0.150              78.5
```

### A/B Test Results
```
✅ Test simulation completed:
   Variant A CTR: 0.050
   Variant B CTR: 0.060
   Lift: 20.0%
   P-value: 0.0234
   Significant: Yes
```

### Content Safety Results
```
✅ Safety check completed:
   Total headlines: 7
   Safe headlines: 5
   Safety rate: 71.4%
   ⚠️ Unsafe: 'This is a fucking amazing deal!' - Content contains profanity
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project directory
   cd "E:\Project\AI Projects\Growth Automation"
   
   # Install missing packages
   pip install -r requirements.txt
   ```

2. **Redis Connection Errors**
   ```bash
   # Redis is optional - the system will work without it
   # To disable Redis, set REDIS_URL="" in .env
   ```

3. **Model Not Found**
   ```bash
   # Train the model first
   python models/train_ctr_model.py
   ```

4. **Permission Errors**
   ```bash
   # On Windows, run as administrator if needed
   # Or use virtual environment
   python -m venv venv
   venv\Scripts\activate
   ```

### Performance Issues

1. **Slow Feature Extraction**
   - Reduce batch size in `api/features.py`
   - Use smaller embedding model

2. **Memory Issues**
   - Reduce training samples in `models/train_ctr_model.py`
   - Use smaller datasets for testing

3. **API Rate Limits**
   - Add delays between requests
   - Use Redis caching

## 📚 Next Steps

### For Development
1. **Customize Prompts**: Edit `prompts/templates.py`
2. **Add Features**: Extend `api/features.py`
3. **Improve Model**: Modify `models/train_ctr_model.py`
4. **Add Tests**: Create tests in `tests/` directory

### For Production
1. **Set Up Monitoring**: Add logging and metrics
2. **Configure CI/CD**: Customize `.github/workflows/ci.yml`
3. **Scale Infrastructure**: Use Kubernetes or cloud services
4. **Add Authentication**: Implement user management

### For Learning
1. **Study the Code**: Read through each module
2. **Run Experiments**: Modify parameters and test
3. **Add Features**: Implement new functionality
4. **Deploy**: Set up production environment

## 🎯 Key Files to Explore

- **`api/generator.py`**: LLM integration and headline generation
- **`api/predictor.py`**: CTR prediction service
- **`experiments/ab_harness.py`**: A/B testing framework
- **`utils/guardrails.py`**: Content safety system
- **`models/train_ctr_model.py`**: Model training pipeline
- **`app/dashboard.py`**: Interactive visualization

## 🚀 Ready to Go!

You now have a complete, production-ready Ad Headline Optimizer system. Start with the basic demos, then explore the advanced features, and finally deploy to production!

---

**Happy optimizing! 🎉**
