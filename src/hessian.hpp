#pragma once
#include <torch/torch.h>
#include <memory>
#include <string>
#include "dataset.hpp"
#include <vector>

class Hessian{
    std::shared_ptr<torch::nn::Module> _model;
    std::string _loss;
    std::shared_ptr<Dataset> _dataset;
    bool _use_cuda;
    int _batch_size;

    Hessian(std::shared_ptr<torch::nn::Module> model,std::string loss, std::shared_ptr<Dataset>dataset,int batch_size, book use_cuda):_model(model), _loss(loss), _dataset(dataset),_use_cuda(use_cuda),_batch_size(batch_size){}

    
   torch::Tensor Compute_Hv(torch::Tensor v){
		std::pair<std::shared_ptr<std::vector<torch::Tensor>>, std::shared_ptr<std::vector<torch::Tensor>>> = _dataset->batch_fetch(_batch_size);
		for(int i = 0 ; i < _batch_size; i++){
			_model->zero_grad();
			
		
		}
		
		
	} 
};
