#!/usr/bin/env python3

#  Raymond Kirk (Tunstill) Copyright (c) 2020
#  Email: ray.tunstill@gmail.com

# Executable for running the detection service server backend from rosrun

import argparse

import rospy

from rasberry_perception import Server, default_service_name


def _default_arg_parser(args=None):
    parser = argparse.ArgumentParser(description='Run the detection server.')
    parser.add_argument('--backend', type=str, help="Which backend to use.", default=None)
    parsed_args, unknown = parser.parse_known_args() #this is an 'internal' method

    # if the server node is launched from using roslaunch
    # these args are prepended. So remove them beforehand.
    unknown = [arg for arg in unknown if "__name:=" not in arg or "__log:=" not in arg]
    
    # Add unrecognised args as kwargs for passing the detection server
    unknown_parser = argparse.ArgumentParser()

    for arg in unknown:
        if arg.startswith(("-", "--")):
            unknown_parser.add_argument(arg, type=str)

    unknown_args = unknown_parser.parse_args(unknown)
    unknown_kwargs = {a: getattr(unknown_args, a) for a in vars(unknown_args)} if len(vars(unknown_args)) else None

    return parsed_args, unknown_kwargs


def __detection_server_runner():
    # Command line arguments should always over ride ros parameters
    args, args_kwargs = _default_arg_parser()

    service_name = default_service_name
    _node_name = service_name + "_server"
    
    rospy.init_node(_node_name)

    server = Server(backend=args.backend, backend_kwargs=args_kwargs, service_name=service_name)
    server.run()


if __name__ == '__main__':
    __detection_server_runner()
