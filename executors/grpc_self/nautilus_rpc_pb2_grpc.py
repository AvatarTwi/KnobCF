# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import executors.grpc_self.nautilus_rpc_pb2 as nautilus__rpc__pb2 # pylint: disable=import-error


class ExecutionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Execute = channel.unary_unary(
                '/ExecutionService/Execute',
                request_serializer=nautilus__rpc__pb2.ExecuteRequest.SerializeToString,
                response_deserializer=nautilus__rpc__pb2.ExecuteReply.FromString,
                )
        self.Heartbeat = channel.unary_unary(
                '/ExecutionService/Heartbeat',
                request_serializer=nautilus__rpc__pb2.EmptyMessage.SerializeToString,
                response_deserializer=nautilus__rpc__pb2.StatsReply.FromString,
                )


class ExecutionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Execute(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Heartbeat(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ExecutionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Execute': grpc.unary_unary_rpc_method_handler(
                    servicer.Execute,
                    request_deserializer=nautilus__rpc__pb2.ExecuteRequest.FromString,
                    response_serializer=nautilus__rpc__pb2.ExecuteReply.SerializeToString,
            ),
            'Heartbeat': grpc.unary_unary_rpc_method_handler(
                    servicer.Heartbeat,
                    request_deserializer=nautilus__rpc__pb2.EmptyMessage.FromString,
                    response_serializer=nautilus__rpc__pb2.StatsReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ExecutionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ExecutionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Execute(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ExecutionService/Execute',
            nautilus__rpc__pb2.ExecuteRequest.SerializeToString,
            nautilus__rpc__pb2.ExecuteReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Heartbeat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ExecutionService/Heartbeat',
            nautilus__rpc__pb2.EmptyMessage.SerializeToString,
            nautilus__rpc__pb2.StatsReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
