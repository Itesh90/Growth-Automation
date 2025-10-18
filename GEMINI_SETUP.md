# 🤖 Gemini API Setup Guide

## 🎯 **What is Gemini?**

Google Gemini is Google's latest AI model that can generate high-quality text, including marketing headlines. It's now integrated into the Ad Headline Optimizer as a third AI option alongside OpenAI and Anthropic.

## 🚀 **Why Use Gemini?**

### **Advantages:**
- **💰 Cost-Effective**: Often cheaper than OpenAI GPT-4
- **⚡ Fast**: Quick response times
- **🎯 High Quality**: Excellent for marketing copy
- **🔄 Reliable**: Google's infrastructure
- **🌍 Multilingual**: Supports many languages

### **Pricing (Approximate):**
- **Input**: $0.50 per 1M characters
- **Output**: $1.50 per 1M characters
- **Example**: 10 headlines ≈ $0.01-0.03

## 🔑 **How to Get Gemini API Key**

### **Step 1: Go to Google AI Studio**
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"

### **Step 2: Create API Key**
1. Choose "Create API Key in new project"
2. Copy the generated API key
3. Keep it secure (don't share publicly)

### **Step 3: Add to Project**
```bash
# Copy environment template
cp env.example .env

# Edit .env file and add:
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🎮 **How to Use Gemini**

### **Option 1: Simple App (Recommended)**
```bash
# Run the simple app
streamlit run app/simple_app.py

# Select "Google Gemini" from AI Model dropdown
```

### **Option 2: Full System**
```bash
# Add API key to .env file
GEMINI_API_KEY=your_key_here

# Run full system
streamlit run app/streamlit_app.py
```

### **Option 3: Test Integration**
```bash
# Test Gemini integration
python test_gemini.py
```

## 🔧 **Configuration Options**

### **Environment Variables:**
```env
# Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=1000
```

### **Available Models:**
- **gemini-1.5-flash**: Fast and efficient (recommended)
- **gemini-1.5-pro**: More capable but slower
- **gemini-1.0-pro**: Stable version

## 🎯 **Usage Examples**

### **Basic Usage:**
```python
from api.generator import HeadlineGenerator, HeadlineRequest

# Initialize generator
generator = HeadlineGenerator()

# Create request
request = HeadlineRequest(
    base_copy="Transform your life with our fitness app",
    tone="Professional",
    n_variants=5
)

# Generate headlines
headlines = await generator.generate_headlines(request)
```

### **Streamlit Integration:**
```python
# In your Streamlit app
if st.button("Generate with Gemini"):
    with st.spinner("Generating with Gemini..."):
        headlines = await generator.generate_headlines(request)
        for headline in headlines:
            st.write(headline.headline)
```

## 📊 **Performance Comparison**

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| **Gemini** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | General use |
| **GPT-4** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Complex tasks |
| **Claude** | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | Creative writing |

## 🛠 **Troubleshooting**

### **Common Issues:**

1. **"No API key configured"**
   ```bash
   # Check .env file
   cat .env | grep GEMINI
   
   # Make sure key is correct
   GEMINI_API_KEY=your_actual_key_here
   ```

2. **"Import error"**
   ```bash
   # Install Gemini library
   pip install google-generativeai
   ```

3. **"Rate limit exceeded"**
   ```bash
   # Wait a few minutes and try again
   # Or reduce number of requests
   ```

4. **"Invalid API key"**
   ```bash
   # Verify key at: https://makersuite.google.com/app/apikey
   # Make sure it's copied correctly
   ```

### **Testing Your Setup:**
```bash
# Test basic integration
python test_gemini.py

# Test with API key
export GEMINI_API_KEY=your_key_here
python test_gemini.py
```

## 🎉 **Success Indicators**

### **✅ Working Correctly:**
- No import errors
- API key accepted
- Headlines generated successfully
- Fast response times (< 5 seconds)

### **❌ Issues to Fix:**
- Import errors → Install dependencies
- API key errors → Check key validity
- Slow responses → Check internet connection
- No output → Check prompt format

## 🚀 **Next Steps**

1. **Get API Key**: Visit https://makersuite.google.com/app/apikey
2. **Add to .env**: Set GEMINI_API_KEY
3. **Test Integration**: Run `python test_gemini.py`
4. **Use in App**: Select Gemini in Streamlit
5. **Generate Headlines**: Create amazing marketing copy!

## 💡 **Pro Tips**

- **Start with gemini-1.5-flash** for best speed/cost ratio
- **Use temperature 0.7** for creative but focused output
- **Test different tones** to see Gemini's versatility
- **Compare with other models** to find your preference
- **Monitor costs** using the built-in cost tracking

**Happy headline generating with Gemini! 🚀**
