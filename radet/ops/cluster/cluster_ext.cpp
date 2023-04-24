#include <torch/extension.h>


std::vector<torch::Tensor> nms_cluster(torch::Tensor &bboxes,
                                       torch::Tensor &scores,
                                       torch::Tensor &labels,
                                       float_t nms_threshold){
    auto order_indices = std::get<1>(torch::sort(scores,0,true));
    auto suppressed = torch::zeros_like(scores, torch::kBool);
    auto instances_id = torch::zeros_like(scores, torch::kInt64);
    auto clusters_num = torch::zeros_like(scores, torch::kInt64);

    auto x1 = bboxes.select(1, 0).contiguous();
    auto y1 = bboxes.select(1, 1).contiguous();
    auto x2 = bboxes.select(1, 2).contiguous();
    auto y2 = bboxes.select(1, 3).contiguous();

    auto order_indices_t = order_indices.data_ptr<int64_t>();
    auto x1_t = x1.data_ptr<float_t>();
    auto y1_t = y1.data_ptr<float_t>();
    auto x2_t = x2.data_ptr<float_t>();
    auto y2_t = y2.data_ptr<float_t>();
    auto labels_t = labels.data_ptr<int64_t>();
    auto suppressed_t = suppressed.data_ptr<bool>();
    auto instances_id_t = instances_id.data_ptr<int64_t>();
    auto clusters_num_t = clusters_num.data_ptr<int64_t>();

    int ndets = scores.size(0);
    int64_t  instance_id = 0;
    int64_t  cluster_num = 0;

    for (int i=0; i<ndets; i++){
        auto index = order_indices_t[i];
        if (suppressed_t[index]){
            continue;
        }
        auto x1_i = x1_t[index];
        auto y1_i = y1_t[index];
        auto x2_i = x2_t[index];
        auto y2_i = y2_t[index];
        auto label_i = labels_t[index];
        auto area_i = (x2_i - x1_i) * (y2_i - y1_i);
        cluster_num = 1;
        suppressed_t[index] = true;
        instances_id[index] = instance_id;


        for (int j=i+1; j< ndets; j++){
            auto index_j = order_indices_t[j];
            auto label_j = labels_t[index_j];
            if (label_j != label_i){
                continue;
            }

            if (suppressed_t[index_j]){
                continue;
            }

            auto x1_j = x1_t[index_j];
            auto y1_j = y1_t[index_j];
            auto x2_j = x2_t[index_j];
            auto y2_j = y2_t[index_j];

            auto x_l = std::max(x1_j, x1_i);
            auto y_t = std::max(y1_j, y1_i);
            auto x_r = std::min(x2_j, x2_i);
            auto y_b = std::min(y2_j, y2_i);
            auto inter_w = std::max(static_cast<float_t>(0), x_r - x_l);
            auto inter_h = std::max(static_cast<float_t>(0), y_b - y_t);
            auto inter = inter_w * inter_h;
            auto area_j = (x2_j - x1_j) * (y2_j - y1_j);
            auto iou = inter /(area_j + area_i - inter);

            if (iou > nms_threshold){
                instances_id_t[index_j] = instance_id;
                suppressed_t[index_j] = true;
                cluster_num ++;
            }

        }
        instances_id_t[index] = instance_id;
        clusters_num_t[index] = cluster_num;
        instance_id ++;

    }
    return {instances_id, clusters_num};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cluster_nms", &nms_cluster, "nms for cluster");
}