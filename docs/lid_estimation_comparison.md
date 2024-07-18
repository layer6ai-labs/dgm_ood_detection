# LID estimation

This document contains information required for running the intrinsic dimension results. Since some of our OOD detection methods rely on fast estimators for intrinsic dimensionality, we have included a section here to actually compare our method for LID estimation.

## LID estimation using Jacobian flows

This method relies on the Jacobian of the flow matching function.

### Sweep for projected mappings

This set of experiments contain datasets that are sampled from a `d` dimensional space and are then projected onto a larger `D` dimensional space. The original noises are either Gaussian or Uniform and they are either projected by repetition (repeating entries an appropriate amount of time) or by using a random linear `D x d` map. Please check the [sweep configuration](../meta_configurations/intrinsic_dimension/projected_flow_fast_lid.yaml) for more information.

```bash
# while being at root 
dysweep_create --config meta_configuration/intrinsic_dimension/projected_flow_fast_lid.yaml
# Run using the following
./meta_run_intrinsic_dimension <sweep-id> 
```

For the baseline, ESS, you may run the following which runs the LID estimator on both the original data as well as the generated samples dataset. You can retrieve the outputs from the `stdout` files created by the sweep.
```bash
# while being at root 
dysweep_create --config meta_configuration/intrinsic_dimension/projected_flow_fast_lid.yaml
# Run using the following
./meta_run_intrinsic_dimension <sweep-id> 
```