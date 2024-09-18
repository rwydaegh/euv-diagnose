# Phase estimator

Use 8,000 training simulations and separate 500-case calibration and test splits. Features are both intensity channels on the fixed wavelength-angle grid. Targets are circular phase offsets at 13.5 nm and 6° relative to the nominal stack. Concentration coefficients vary within ±3% of nominal, intersected with source bounds. Noise scales are 1e-4 and 1e-3; intensity gains vary from 0.98 to 1.02 and angle shifts from −0.1 to 0.1°.

The nominal neural RMSE is 2.40°, versus 5.22° for ridge. Nominal 90% intervals cover 89.4%. Widening concentration ranges to ±8% while adding correlated wavelength error lowers coverage to 74.6%. These two changes are simultaneous; this does not isolate their effects. Experimental phase accuracy has not been measured.
