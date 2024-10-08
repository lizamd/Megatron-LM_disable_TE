#!/usr/bin/env bash
# Author: Scott Moe

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo " "
  echo "*********************************************************************************************"
  echo "Babelstream build helper script usage:"
  echo "./build_babelstream.sh "
  echo "*********************************************************************************************"
  echo " "
  echo "    This script builds and runs BabelStream. Users should provide inputs from the list below:"
  echo " "
  echo "    [-h|--help] prints the help message"
  echo "    [--with-rocm=<dir>] Path to ROCm install (Default: $ROCM_PATH if defined or /opt/rocm)"
  echo "    [--build=<option>] Build BabelStream using one of two commands."
  echo "           --build=1 executes [ELTS_PER_LANE=1 CHUNKS_PER_BLOCK=1 make -f HIP.make] (recommended for MI100)"
  echo "           --build=2 executes [make -f HIP.make]"
  echo "    [--run-problem=#] Run one of the two standard BabelStream test problems. Passing all runs all tests"
  echo "           --run-problem=1 tests the bandwidth using fp64"
  echo "           --run-problem=2 tests the bandwidth using fp32"
  echo " "
  echo "    For example [./build_babelstream.sh --build=1] clones and builds babelstream,"
  echo "    [./build_babelstream.sh --run-problem=1] just runs the first test problem, and"
  echo "    [./build_babelstream.sh --build=1 --run-problem=1] builds and runs the first test problem."
  echo " "
}


# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: Fine
# Exit code 1: problems with getopt



# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

## #################################################
## Parameter parsing
## #################################################
#
# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,with-rocm:,build:,run-problem: --options h -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

B_build=-1
run_problem=-1

case "${1}" in
    --)
        display_help
	exit 1
esac

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    --with-rocm)
        with_rocm=${2}
        shift 2 ;;
    --build)
	B_build=${2}
	shift 2 ;;
    -r|--run-problem)
	run_problem=${2}
	shift 2 ;;
    --)
	shift ;
	break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done



if [[ ${B_build} -gt -1 ]]; then
   case ${B_build} in
        1)
              make clean -f HIP.make
	      echo "running ELTS_PER_LANE=1 CHUNKS_PER_BLOCK=1 make -f HIP.make"
	      ELTS_PER_LANE=1 CHUNKS_PER_BLOCK=1 make -f HIP.make
	      ;;
        2)
              make clean -f HIP.make
              echo "running make -f HIP.make"
              make -f HIP.make
	      ;;
	*)
	      echo "Unrecognized option! Recognized options are:"
	      echo "1) ELTS_PER_LANE=1 CHUNKS_PER_BLOCK=1 make -f HIP.make"
	      echo "2) make -f HIP.make"
     esac
fi


if [[ ${run_problem} -gt -1 ]]; then
   echo "run_problem= " ${run_problem}

   case ${run_problem} in
       1)
	  ./hip-stream -e --std -s $((256 * 1024 * 1024))
         ;;
       2)
	  ./hip-stream -e --std -s $((256 * 1024 * 1024)) --float
         ;;
       all)
	  ./hip-stream -e --std -s $((256 * 1024 * 1024))
	  ./hip-stream -e --std -s $((256 * 1024 * 1024)) --float
	 ;;
       *)  echo "This problem number doesn't exist. Available problems are:";
	      echo "1) ./hip-stream -e --std -s $((256 * 1024 * 1024))"
	      echo "2) ./hip-stream -e --std -s $((256 * 1024 * 1024)) --float"
	      echo "all) executes all of the above"
        exit 1
        ;;
   esac
fi
