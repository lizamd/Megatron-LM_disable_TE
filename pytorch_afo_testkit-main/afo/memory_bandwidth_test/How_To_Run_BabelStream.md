# How to Build and Run `Babel Stream` 
 
## Step 1: Build `Babel Stream`
Clone the repository:
```ruby
git clone https://github.com/ROCm/pytorch_afo_testkit.git
cd pytorch_afo_testkit
```

### Build `Babel Stream` Docker Image
The `memory_bandwidth_test/Dockerfile` can be used to build `Babel Stream` test as
```ruby
docker build --no-cache -t rocm/afo:latest -f memory_bandwidth_test/Dockerfile .
```

## Step 2: How to Run The `Babel Stream` Test

Script `memory_bandwidth_test/run_babel_stream.sh` runs the `Babel Stream` test. The test can be executed from "/rocm" directory inside the docker image.

```ruby
cd /rocm
bash /rocm/memory_bandwidth_test/run_babel_stream.sh
```

The performance number (the average of `triad`, `copy`, `read`) of `Babel Stream` test will be displayed at the end of running the test in terms of `MiB/s`.
