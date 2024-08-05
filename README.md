# Hyper-rectangular clustering with outliers
## This is not an officially supported Google product

Library to find the optimal hyper-rectangular clustering assuming a maximum
number of outliers and a maximum number of clusters. This code solve the problem
using a Branch-and-Price formulation of the model.

The code uses SCIP for solving the problem, with different pricer for
generating new variables:

- Max-Flow algorithm (only for solving the LP relaxation of the problem)
- CpSat (a constraint-programming solver)
- MathOpt (a MIP formulation)

All these libraries are provided by Google Optimization Tools (a.k.a.,
OR-Tools).

## Installing
This library requires:

- CMake (https://cmake.org)
- Abseil (https://abseil.io)
- Ortools (https://developers.google.com/optimization)
- A MIP solver supported by Ortools::MathOpt. Currently coded also using Scip.
It can be replaced on `pricer/pricer_mip/pricer_mip.h` for other solvers
supported by mathopt (in particular, `math_opt::SolverType::kGurobi`).

Also, CMake will download GoogleTest for unit testing.

## Usage
After compiling, the main binary is placed in `bin/` directory. For information
usage, execute `problem_main --help`

```
  Flags from Users/emoreno/Code/clusteringSquare/src/master_scip_main.cc:
    --branching (Branching to use [none|ryanfoster]); default: ryanfoster;
    --input_data_file (Input file with problem data); default: "";
    --pricer (Pricer to use [cpsat|cpsat_enforcing,maxclosure,mip]);
      default: cpsat;
    --relax_coverage_vars (Relax integrality of coverage vars); default: false;
    --relax_priced_vars (Relax integrality of priced vars); default: false;
    --time_limit (time limit (in seconds)); default: 300;
```
By default, it solves the problem using Branch-and-Price with RyanFoster
branching rule. Note: `--pricer=maxclosure` cannot be used with these default
parameters. For solving the LP-relaxation of the problem using column
generation, you must set 
`--branching=none --relax_coverage_vars --relax_priced_vars`. In this case
maxclosure is a valid pricer for the problem.


## Input files
The format of the input is a tab-separated text file with the following lines:

- First line: `<number of points>` `<number of dimensions>`
`<max number of clusters>` `<max number of outliers>`
- i-th line: Coordinates of the i-th point. One **integer** value for dimension.

Example:

```
6      2       2      1
-6986  663162
-6164  350438
-5576  5658
19626  1056136
23130  -812623
13931  -692002
```



