import numpy
import joblib
import json
from azureml.core.model import Model

# from inference_schema.schema_decorators import input_schema, output_schema


def init():
    # load the model from file into a global object
    global model

    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = Model.get_model_path(
        model_name="driver_training_model.pkl")
    model = joblib.load(model_path)


# input_sample = numpy.array([
#     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
#     [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
# output_sample = numpy.array([
#     5021.509689995557,
#     3693.645386402646])


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
# @input_schema('data', NumpyParameterType(input_sample))
# @output_schema(NumpyParameterType(output_sample))
def run(raw_data, request_headers):

    
    data = json.loads(raw_data)["data"]
    data = numpy.array(data) #Ensuring that we're deserialising the data in str format
    result = model.predict(data)

    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'traceparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
    print(
        (
            '{{"RequestId":"{0}", '
            '"TraceParent":"{1}", '
            '"NumberOfPredictions":{2}}}'
        ).format(
            request_headers.get("X-Ms-Request-Id", ""),
            request_headers.get("Traceparent", ""),
            len(result),
        )
    )

    return {"result": result.tolist()}


if __name__ == "__main__":
    # Test scoring
    init()
    test_row = '{"data": [[0,1,8,1,0,0,1,0,0,0,0,0,0,0,12,1,0,0,0.5,0.3,0.610327781,7,1,-1,0,-1,1,1,1,2,1,65,1,0.316227766,0.669556409,0.352136337,3.464101615,0.1,0.8,0.6,1,1,6,3,6,2,9,1,1,1,12,0,1,1,0,0,1],[4,2,5,1,0,0,0,0,1,0,0,0,0,0,5,1,0,0,0.9,0.5,0.771362431,4,1,-1,0,0,11,1,1,0,1,103,1,0.316227766,0.60632002,0.358329457,2.828427125,0.4,0.5,0.4,3,3,8,4,10,2,7,2,0,3,10,0,0,1,1,0,1]]}'
    prediction = run(test_row, {})
    print("Test result: ", prediction)
