# AI-Orchestrated PowerPoint Report Generator

## Overview

This notebook leverages your **LLM API's agentic capabilities** (ReAct agent + python_coder tool) to automatically generate comprehensive PowerPoint reports from warpage statistics data.

## Key Advantage

Instead of **2000+ lines of hardcoded visualization logic**, this approach uses **~200 lines of strategic prompts** to let your AI agent autonomously:
- Analyze data
- Generate visualizations
- Create PowerPoint slides
- Handle errors automatically

## Files

### **PPTX_Report_Generator_Agent.ipynb** (NEW - Recommended)
**AI-Orchestrated Approach** - Minimal code, maximum automation
- 7 cells total
- Sends strategic prompts to your LLM API
- Agent handles all code generation and execution
- Self-healing (auto-fixes errors)
- Supports multiple datasets
- **Adaptive to data structure**

### **PPTX_Report_Generator.ipynb** (Original)
**Hardcoded Approach** - For reference/fallback
- 47 cells total
- All visualization code pre-written
- Manual error handling
- Works offline
- **Fixed visualization logic**

## How to Use (Agent Version)

### 1. Configure Data Files
```python
# Edit cell in Section 2
stats_paths = [
    Path("data/uploads/leesihun/20251013_stats.json"),
    # Path("data/uploads/leesihun/20251014_stats.json"),  # Add more for comparison
]
```

### 2. Run All Cells
The notebook will automatically:
1. **Phase 1**: Upload files → AI analyzes data → Returns insights
2. **Phase 2**: AI generates 8-9 visualizations → Saves to temp_charts/
3. **Phase 3**: AI creates PowerPoint → Adds all charts → Saves .pptx

### 3. Get Your Report
Output: `Warpage_Report_YYYYMMDD_HHMMSS.pptx` with 11-12 slides

## What the AI Does

### Phase 1: Data Analysis
```
User uploads JSON files
    ↓
ReAct Agent analyzes
    ↓
python_coder executes analysis code
    ↓
Returns: Statistics, outliers, visualization recommendations
```

### Phase 2: Visualization Generation
```
User sends chart specifications
    ↓
Agent generates matplotlib/seaborn code
    ↓
python_coder executes → Saves 8-9 PNGs
    ↓
Returns: Confirmation + chart descriptions
```

### Phase 3: PowerPoint Assembly
```
User sends slide specifications
    ↓
Agent generates python-pptx code
    ↓
python_coder creates slides + adds charts
    ↓
Returns: Saved .pptx file path
```

## Strategic Prompts

The notebook uses 3 carefully crafted prompts:

1. **Analysis Prompt** (~300 words)
   - Requests data loading and combination
   - Asks for key statistics
   - Requests visualization recommendations

2. **Visualization Prompt** (~500 words)
   - Specifies 8-9 charts with exact requirements
   - Provides technical specs (DPI, colors, layout)
   - Lists required packages

3. **PowerPoint Prompt** (~600 words)
   - Defines slide structure (11-12 slides)
   - Specifies formatting for each slide
   - Provides exact dimensions and colors

## Architecture

```
Jupyter Notebook (Orchestrator)
        ↓
    LLM API Client
        ↓
    ReAct Agent
        ↓
    python_coder Tool
        ↓
    [Code Generation] → [Verification] → [Execution] → [Retry if error]
        ↓
    Result (Charts + PowerPoint)
```

## Benefits vs Hardcoded Approach

| Feature | AI-Orchestrated | Hardcoded |
|---------|----------------|-----------|
| **Lines of code** | ~200 | 2000+ |
| **Adaptability** | High (AI adjusts) | Low (fixed logic) |
| **Error handling** | Auto-fix (5 retries) | Manual |
| **Multi-file support** | Built-in | Built-in |
| **Customization** | Edit prompts | Edit code |
| **Maintenance** | Low | High |
| **Dependencies** | API + python_coder | All packages |

## Requirements

### For AI-Orchestrated Version:
- Running LLM API server (http://localhost:1007)
- Valid credentials (username/password)
- ReAct agent enabled
- python_coder tool available
- Data files in `data/uploads/{username}/`

### For Hardcoded Version:
- All packages: `pip install python-pptx matplotlib seaborn pandas numpy scipy scikit-learn`

## Troubleshooting

### AI-Orchestrated Version

**Problem**: Agent doesn't generate code
- **Solution**: Check agent_type="auto" is set, or force "react"

**Problem**: Code execution fails
- **Solution**: Agent auto-retries up to 5 times. Check error message in output.

**Problem**: Charts not generated
- **Solution**: Check temp_charts/ directory. Agent will report errors.

**Problem**: PowerPoint not created
- **Solution**: Ensure Phase 2 completed successfully. Check file paths in prompts.

### General

**Problem**: Files not found
- **Solution**: Verify paths in Section 2. Check file permissions.

**Problem**: API timeout
- **Solution**: Increase timeout in client initialization (default 3600s)

**Problem**: No model available
- **Solution**: Check API server is running, login successful

## Example Output

```
Phase 1: Data Analysis (30-60s)
  ✓ Loaded 50 measurements
  ✓ Calculated statistics
  ✓ Recommended 8 visualizations

Phase 2: Visualization Generation (60-120s)
  ✓ temporal_trends.png
  ✓ distributions.png
  ✓ boxplots.png
  ✓ pca_scatter.png
  ✓ correlation_heatmap.png
  ✓ control_chart.png
  ✓ radar_chart.png
  ✓ summary_table.png

Phase 3: PowerPoint Assembly (30-60s)
  ✓ Created 11 slides
  ✓ Added all charts
  ✓ Formatted professionally
  ✓ Saved: Warpage_Report_20250119_143052.pptx (2.4 MB)

Total time: ~3 minutes
```

## Customization

### Add More Charts
Edit the visualization prompt in Section 4:
```python
# Add to visualization_prompt
10. **Custom Chart** (custom_chart.png)
   - Description of what to visualize
   - Technical specifications
```

### Change PowerPoint Style
Edit the PowerPoint prompt in Section 5:
```python
# Modify colors, fonts, layout
- RGB colors: Blue (31, 119, 180) → Your color
- Font sizes: 32pt → Your size
```

### Adjust Analysis
Edit the analysis prompt in Section 3:
```python
# Add custom analysis requirements
5. **Custom Analysis:**
   - Your specific requirements
```

## Advanced Usage

### Multi-Dataset Comparison
```python
stats_paths = [
    Path("data/uploads/leesihun/batch1_stats.json"),
    Path("data/uploads/leesihun/batch2_stats.json"),
    Path("data/uploads/leesihun/batch3_stats.json"),
]
```
Agent automatically generates dataset comparison chart.

### Continuous Conversation
All phases use the same session_id, so the agent maintains context:
```python
# Phase 1
analysis, session_id = client.chat_new(...)

# Phase 2 (remembers Phase 1)
viz, _ = client.chat_continue(MODEL, session_id, ...)

# Phase 3 (remembers Phase 1 & 2)
pptx, _ = client.chat_continue(MODEL, session_id, ...)
```

### Error Recovery
If a phase fails, you can retry from that phase:
```python
# Re-run just the failed cell
viz_result, _ = client.chat_continue(MODEL, session_id, visualization_prompt)
```

## Performance Tips

1. **Use faster model for quick iterations**
   - Switch to lighter model during testing
   - Use full model for production

2. **Reduce chart count for faster generation**
   - Comment out charts you don't need
   - Test with 2-3 charts first

3. **Reuse session for multiple requests**
   - Agent remembers data context
   - Faster subsequent requests

## Version History

- **v1.0.0** (January 2025): Initial AI-orchestrated version
  - 3-phase autonomous generation
  - Strategic prompt engineering
  - Multi-file support
  - Self-healing execution

## Future Enhancements

Potential improvements:
1. Add interactive chart selection
2. Support for additional data formats (Excel, CSV)
3. Custom template support
4. Real-time streaming updates
5. Export to PDF option

## Support

For issues with:
- **API connection**: Check backend server logs
- **Code generation**: Review agent execution trace
- **File paths**: Verify data directory structure
- **Package errors**: Check python_coder tool configuration

## Related Files

- `API_examples.ipynb`: API usage examples
- `CLAUDE.md`: Project documentation
- `backend/tasks/react/`: ReAct agent implementation
- `backend/tools/python_coder/`: Code generation tool

---

**Recommendation**: Use the AI-orchestrated version for most cases. It's more flexible and requires less maintenance.
