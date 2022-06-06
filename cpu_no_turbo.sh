# Tensorflow 2.9.1 training reboots my machine with a nvidia 2070 super
# This seems to fix it (TODO: Confirm it)
# https://github.com/tensorflow/tensorflow/issues/8858
# https://askubuntu.com/questions/619875/disabling-intel-turbo-boost-in-ubuntu
echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

