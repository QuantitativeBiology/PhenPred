# Weight Calculation Documentation for Cell Line Selection

## 🧮 Weight Calculation Deep Dive

### 🎯 Purpose of Weighting
The weight calculation ensures that your cell line selection prioritizes the most **statistically significant** synthetic lethality relationships discovered by the VAE analysis, while still considering all gene pairs.

The core principle: **Lower FDR values (more significant) → Higher weights (higher priority)**

---

## 📊 Mathematical Transformation Pipeline

### Step 1: -log10 Transformation
```python
log_weights = -log10(FDR_values)
```
- **Why**: Converts small FDR values (good) to large numbers (high priority)
- **Example**: FDR = 1e-180 → -log10(1e-180) = 180
- **Result**: Higher values = more significant relationships

### Step 2: Normalization [0,1]
```python
normalized = (log_weights - min) / (max - min)
```
- **Why**: Standardizes the range regardless of extreme FDR values
- **Example**: [180, 134, 50, 2] → [1.0, 0.75, 0.27, 0.0]
- **Result**: Relative rankings preserved in standard range

### Step 3: Power Transformation
```python
powered = normalized^power  # Default power = 2.0
```
- **Why**: Amplifies differences between top and bottom pairs
- **Example**: [1.0, 0.75, 0.27, 0.0] → [1.0, 0.56, 0.07, 0.0]
- **Result**: Top pairs get disproportionately higher weights

### Step 4: Range Scaling
```python
final_weight = 1.0 + (5.0 - 1.0) × powered
```
- **Why**: Maps to interpretable weight range [1.0, 5.0]
- **Example**: [1.0, 0.56, 0.07, 0.0] → [5.0, 3.24, 1.28, 1.0]
- **Result**: Most significant pairs get 5× priority over least significant

---

## 🔋 Power Parameter Effects

| Power Value | Distribution | Impact | Weight Distribution Example |
|-------------|--------------|---------|---------------------------|
| **1.0** | Linear | Equal weight differences | 1.0 → 1.8 → 2.6 → 3.4 → 4.2 → 5.0 |
| **2.0** (default) | Quadratic | Moderate top-heavy | 1.0 → 1.16 → 1.64 → 2.44 → 3.56 → 5.0 |
| **3.0** | Cubic | Strong top-heavy | 1.0 → 1.03 → 1.26 → 1.86 → 3.05 → 5.0 |

### Visual Comparison (6 gene pairs ranked by significance)
- **Linear (power=1)**: Evenly spaced weights across significance levels
- **Quadratic (power=2)**: Moderate emphasis on most significant pairs  
- **Cubic (power=3)**: Strong emphasis on top pairs, minimal weight for others

---

## 🎯 Integration with Selection Algorithm

### Weighted Scoring Formula
```python
Weighted_Score = Validation_Score × FDR_Weight
```

### Coverage Calculation Formula
```python
Weighted_Coverage = Σ(FDR_weight × coverage) for all gene pairs
```

### Selection Priority Logic
1. Cell lines covering high-FDR-weight pairs get selected first
2. Ensures experimental validation focuses on strongest statistical evidence
3. Balances comprehensive coverage with significance prioritization

---

## 📈 Real Example Transformations

### High Significance Gene Pair
- **Input FDR**: 1e-180 (extremely significant)
- **-log10 Transform**: 180
- **After Normalization**: ~1.0 (top of range)
- **After Power^2**: 1.0 (unchanged)
- **Final Weight**: ~5.0 (maximum priority)

### Medium Significance Gene Pair
- **Input FDR**: 1e-50 (moderately significant)
- **-log10 Transform**: 50
- **After Normalization**: ~0.75
- **After Power^2**: 0.56
- **Final Weight**: ~3.24 (moderate priority)

### Low Significance Gene Pair
- **Input FDR**: 0.01 (marginally significant)
- **-log10 Transform**: 2
- **After Normalization**: ~0.0 (bottom of range)
- **After Power^2**: 0.0 (unchanged)
- **Final Weight**: ~1.0 (minimum priority)

---

## ✅ Why This Approach Works

### 🎪 Statistical Rigor
- **Direct Connection**: Discovery significance directly influences validation priority
- **Evidence Hierarchy**: Prevents equal treatment of strong vs. weak statistical evidence
- **Scientific Principle**: Maintains established hierarchy of statistical evidence

### ⚖️ Balanced Optimization
- **Inclusive Approach**: All gene pairs still considered (minimum weight = 1.0)
- **Proportional Boost**: Top pairs get appropriate boost (maximum weight = 5.0)
- **No Exclusion**: Avoids complete exclusion of lower-significance pairs

### 🔧 Tunable Parameters
- **min_weight/max_weight**: Controls overall weight range
- **power**: Controls emphasis strength on top pairs
- **Easy Adjustment**: Can modify based on experimental capacity and preferences

### 📊 Interpretable Results
- **Clear Relationship**: Transparent Lower FDR → Higher Weight mapping
- **Mathematical Transparency**: Well-documented transformation process
- **Audit Trail**: Complete path from FDR values to final selection

---

## 🚀 Practical Impact on Cell Line Selection

### Weight Distribution in Practice
When you run the selection algorithm:

1. **Gene pairs with FDR ~ 1e-180** get weight ≈ 5.0 (highest priority)
2. **Gene pairs with FDR ~ 1e-100** get weight ≈ 4.2 (very high priority)
3. **Gene pairs with FDR ~ 1e-50** get weight ≈ 3.0 (moderate priority)  
4. **Gene pairs with FDR ~ 1e-10** get weight ≈ 1.8 (lower priority)
5. **Gene pairs with FDR ~ 0.01** get weight ≈ 1.0 (minimum priority)

### Selection Outcome
Your selected cell lines will naturally demonstrate the **most statistically robust** synthetic lethality relationships, while still providing some coverage of the broader set of candidates.

### Experimental Efficiency
- **Resource Optimization**: Limited experimental capacity directed to best candidates
- **Success Probability**: Maximizes likelihood of successful validation
- **Reduced Risk**: Focus on relationships with strongest statistical support

---

## 🔧 Implementation Details

### Function Signature
```python
def calculate_principled_weights(fdr_values, min_weight=1.0, max_weight=5.0, power=2.0):
    """
    Calculate weights from FDR values with clear parameterization.

    Args:
        fdr_values: Array of FDR p-values
        min_weight: Minimum weight for least significant pairs (default=1.0)
        max_weight: Maximum weight for most significant pairs (default=5.0)
        power: Exponent > 1 to favor top pairs more strongly (default=2.0)

    Returns:
        Array of weights where lower FDR gets higher weight
    """
```

### Safety Features
- **Zero Protection**: `max(fdr_values, 1e-100)` prevents log(0) errors
- **Uniform Handling**: When all FDR values identical, assigns uniform weights
- **Numerical Stability**: Robust to extreme FDR values

### Parameter Recommendations
- **Default Settings**: `min_weight=1.0, max_weight=5.0, power=2.0`
- **Conservative**: `power=1.5` for less aggressive top-pair emphasis
- **Aggressive**: `power=3.0` for stronger focus on most significant pairs
- **Custom Range**: Adjust `min_weight/max_weight` based on desired priority spread

---

## 📊 Quality Control and Validation

### Sanity Checks
1. **Monotonicity**: Verify lower FDR → higher weight always holds
2. **Range Bounds**: Confirm all weights fall within [min_weight, max_weight]
3. **Distribution**: Check weight distribution matches expected emphasis level

### Diagnostic Outputs
- **Weight Statistics**: Min, max, mean, std of calculated weights
- **FDR-Weight Correlation**: Should be strongly negative (Pearson r ≈ -1)
- **Top/Bottom Ratios**: Verify reasonable separation between significance levels

---

## 🎯 Summary

The weight calculation transforms FDR values into meaningful selection priorities through a principled mathematical approach:

1. **Statistical Foundation**: Based on -log10 p-value transformation (standard in genomics)
2. **Controllable Emphasis**: Power parameter allows tuning of top-pair preference
3. **Balanced Coverage**: All pairs retained but with appropriate prioritization
4. **Experimental Efficiency**: Directs limited resources to most promising candidates

This approach ensures that your **cell line selection is both statistically principled and experimentally practical**, maximizing the likelihood of successful synthetic lethality validation while maintaining comprehensive coverage of your discovery results.

---

## 📚 References and Related Concepts

### Statistical Background
- **-log10 p-value**: Standard transformation in genomics for significance visualization
- **Power Law**: Mathematical relationship for emphasis control
- **Min-Max Normalization**: Standard preprocessing technique in machine learning

### Biological Context
- **FDR Control**: Benjamini-Hochberg correction for multiple testing
- **Synthetic Lethality**: Biological concept underlying the validation approach
- **VAE Discovery**: Variational autoencoder analysis generating the input FDR values

### Algorithmic Context
- **Greedy Selection**: Algorithm using these weights for cell line selection
- **Weighted Coverage**: Coverage metric incorporating statistical significance
- **Multi-objective Optimization**: Balancing coverage breadth with significance depth 