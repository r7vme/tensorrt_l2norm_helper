#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <NvInfer.h>
#include <NvUffParser.h>
#include <NvInferPlugin.h>
#include <l2norm_helper.h>

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;

class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
      cout << msg << endl;
  }
} gLogger;


int main()
{
  // Register default TRT plugins (e.g. LRelu_TRT)
  if (!initLibNvInferPlugins(&gLogger, "")) { return 1; }

  /* parse uff */
  IBuilder *builder = createInferBuilder(gLogger);
  INetworkDefinition *network = builder->createNetwork();
  IUffParser *parser = createUffParser();

  // USER DEFINED VALUES
  string uffFile="sample.uff";
  string engineFile="sample.engine";
  DataType dtype = DataType::kFLOAT;
  parser->registerInput("input_1", DimsCHW(3, 224, 224), UffInputOrder::kNCHW);
  parser->registerOutput("orientation/l2_normalize");
  parser->registerOutput("dimension/LeakyRelu");
  parser->registerOutput("confidence/Softmax");
  // END USER DEFINED VALUES

  if (!parser->parse(uffFile.c_str(), *network, dtype))
  {
    cout << "Failed to parse UFF\n";
    builder->destroy();
    parser->destroy();
    network->destroy();
    return 1;
  }

  /* build engine */
  if (dtype == DataType::kHALF)
  {
    builder->setFp16Mode(true);
  }
  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(1<<30);
  // Disable DLA, because many layers are still not supported
  // and this causes additional latency.
  //builder->allowGPUFallback(true);
  //builder->setDefaultDeviceType(DeviceType::kDLA);
  //builder->setDLACore(1);
  ICudaEngine *engine = builder->buildCudaEngine(*network);

  /* serialize engine and write to file */
  ofstream planFile;
  planFile.open(engineFile);
  IHostMemory *serializedEngine = engine->serialize();
  planFile.write((char *)serializedEngine->data(), serializedEngine->size());
  planFile.close();

  /* break down */
  builder->destroy();
  parser->destroy();
  network->destroy();
  engine->destroy();
  serializedEngine->destroy();

  return 0;
}
