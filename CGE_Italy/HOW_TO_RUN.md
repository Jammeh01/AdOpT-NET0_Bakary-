# How to Run the CGE Italy Model

## Quick Start

Your CGE model for Italy is now fully set up and ready to run! All files will be created and saved within the `CGE_Italy` folder as requested.

### 1. Navigate to the CGE_Italy folder

The model is located in:

```
C:\Users\Jamme002\OneDrive - Universiteit Utrecht\Documents\AdOpT-NET0_Bakary-\CGE_Italy\
```

### 2. Run the Model

**Option A: Run all scenarios at once (recommended)**

```powershell
cd "C:\Users\Jamme002\OneDrive - Universiteit Utrecht\Documents\AdOpT-NET0_Bakary-\CGE_Italy"
& "C:/Users/Jamme002/OneDrive - Universiteit Utrecht/Documents/AdOpT-NET0_Bakary-/.venv/Scripts/python.exe" main.py
```

**Option B: Run individual scenarios**

```powershell
cd "C:\Users\Jamme002\OneDrive - Universiteit Utrecht\Documents\AdOpT-NET0_Bakary-\CGE_Italy"
& "C:/Users/Jamme002/OneDrive - Universiteit Utrecht/Documents/AdOpT-NET0_Bakary-/.venv/Scripts/python.exe" run_model.py baseline
& "C:/Users/Jamme002/OneDrive - Universiteit Utrecht/Documents/AdOpT-NET0_Bakary-/.venv/Scripts/python.exe" run_model.py moderate
& "C:/Users/Jamme002/OneDrive - Universiteit Utrecht/Documents/AdOpT-NET0_Bakary-/.venv/Scripts/python.exe" run_model.py ambitious
```

### 3. View Results

**Interactive results viewer:**

```powershell
& "C:/Users/Jamme002/OneDrive - Universiteit Utrecht/Documents/AdOpT-NET0_Bakary-/.venv/Scripts/python.exe" view_results.py
```

## Model Output

All results are saved in `CGE_Italy/results/` folder:

- **JSON files**: Detailed model results with full data structures
- **Excel files**: Summary tables with key economic indicators

## Model Configuration

- **Base Year**: 2021
- **Final Year**: 2050 (configurable)
- **Sectors**: 11 economic sectors including energy, transport, and other sectors
- **Regions**: 5 Italian NUTS-1 regions (Northwest, Northeast, Centre, South, Islands)
- **Base GDP**: €1,782,000 million

## Scenarios Explained

1. **Baseline**: Current policies continuation with moderate carbon pricing
2. **Moderate**: Enhanced carbon pricing and energy efficiency policies
3. **Ambitious**: Strong carbon pricing and rapid decarbonization policies

## Key Features

✅ **Folder Isolation**: All outputs stay within CGE_Italy folder  
✅ **Timestamped Results**: Results files include timestamps to prevent overwriting  
✅ **Multiple Formats**: Results available in both JSON and Excel formats  
✅ **Comprehensive Logging**: Detailed console output during execution  
✅ **Error Handling**: Graceful error handling with informative messages  
✅ **Flexible Execution**: Multiple ways to run the model (all scenarios or individual)

## Next Steps

1. Run the model using one of the methods above
2. Check the `results/` folder for your output files
3. Use `view_results.py` to explore and analyze the results
4. Modify scenarios or parameters in `main.py` if needed

The model is now ready to use! 🚀
