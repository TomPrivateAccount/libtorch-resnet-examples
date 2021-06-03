#include <torch/torch.h>
#include "resnet.hpp"
#include "dataset.hpp"
#include <memory>
#include <iostream>

void model(){

 torch::Device device("cpu");
  if (torch::cuda::is_available()){
    std::cout<<"using cuda"<<std::endl;
    device = torch::Device("cuda:0");
  }

  torch::Tensor input = torch::randn({2, 3, 224, 224}).to(device);
  std::cout<<"build net"<<std::endl;
  ResNet<BasicBlock> resnet = resnet18();
  std::cout<<"net to device"<<std::endl;
  resnet.to(device);
 

  torch::optim::Adam opt(resnet.parameters(),torch::optim::AdamOptions(0.001));

  torch::Tensor target = torch::randn({2,1000}).to(device);

  for (size_t i = 0; i < 4; i++) {
    std::cout<<"forward begin"<<std::endl<<std::flush;
    torch::Tensor output = resnet.forward(input);
    std::cout<<"forward end "<<output.sizes()<<std::endl<<std::flush;
    auto loss = torch::mse_loss(output.view({2,1000}), target);
    std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
    loss.backward();
    opt.step();
  }

  std::cout<<"forward net"<<std::endl;
  input = resnet.forward(input);
  std::cout << input.sizes() << std::endl;
}



void dataset(){
  CIFAR10_Dataset cifar("/home/chengdaning/data/cifar-10-binary/cifar-10-batches-bin/", false); 

  std::shared_ptr<torch::Tensor> data ;
  std::shared_ptr<torch::Tensor> label;

  cifar.GetBatchInTensor(2,data,label);

  std::cout<<*data<<std::endl;
  std::cout<<"-----------------------------------------------"<<std::endl;
  std::cout<<*label<<std::endl;
}


void forward_cpu(){
  torch::Device device("cpu");

  CIFAR10_Dataset cifar("/home/chengdaning/data/cifar-10-binary/cifar-10-batches-bin/", false); 

  std::shared_ptr<torch::Tensor> input;
  std::shared_ptr<torch::Tensor> target;

  std::cout<<"build net"<<std::endl;
  ResNet<BasicBlock> resnet = resnet18_cifar10();
  std::cout<<"net to device"<<std::endl;

  torch::optim::Adam opt(resnet.parameters(),torch::optim::AdamOptions(0.001));

  for(int i = 0 ; i < 5 ; i++){
    cifar.GetBatchInTensor(2,input,target);

     
    std::cout<<"forward begin"<<std::endl<<std::flush;
    torch::Tensor output = resnet.forward(*input);
    std::cout<<"forward end "<<output.sizes()<<std::endl<<std::flush;
    auto loss = torch::mse_loss(output.view({2,10}), *target);
    std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
    loss.backward();
    opt.step();
  }
}

void forward_gpu(){
  torch::Device device("cpu");
  if (torch::cuda::is_available()){
    std::cout<<"using cuda"<<std::endl;
    device = torch::Device("cuda:0");
  }

  CIFAR10_Dataset cifar("/home/chengdaning/data/cifar-10-binary/cifar-10-batches-bin/", false); 

  std::shared_ptr<torch::Tensor> input;
  std::shared_ptr<torch::Tensor> target;

  std::cout<<"build net"<<std::endl;
  ResNet<BasicBlock> resnet = resnet18_cifar10();
  std::cout<<"net to device"<<std::endl;
  resnet.to(device);
  torch::optim::Adam opt(resnet.parameters(),torch::optim::AdamOptions(0.001));

  for(int i = 0 ; i < 10000 ; i++){
    cifar.GetBatchInTensor(2,input,target);
    //std::cout<<"forward begin"<<std::endl<<std::flush;
    torch::Tensor output = resnet.forward(input->to(device));
    //std::cout<<"forward end "<<output.sizes()<<std::endl<<std::flush;
    auto loss = torch::mse_loss(output.view({2,10}), target->to(device));
    std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
    loss.backward();
    opt.step();
  }
}

int main(){
  model();
  dataset();
  forward_gpu();
  forward_gpu();
}