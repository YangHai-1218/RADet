#include <torch/extension.h>
#include "cmath"
#include <numeric>
#include <queue>



float_t vote_single_dim(std::vector<float_t> &scores, std::vector<float_t> &x){
//  pre compute
    float_t score_sum = 0;
    float_t  voted_x = 0;
    int n = scores.size();
    for (int i=0; i < n; i++){
        score_sum += scores[i];
        voted_x += scores[i] * x[i];
    }
    voted_x = voted_x / score_sum;
//  filter
    float_t sigma_x = 0;
    for (int i=0; i < n; i++){
        sigma_x += scores[i] * (x[i] - voted_x) * (x[i] - voted_x);
    }
    sigma_x = sqrt(sigma_x / score_sum);
//  final compute
    float_t filter_score_sum = 0.;
    float_t final_voted_x = 0.;
    for (int i=0; i < n; i++) {
        if ((voted_x - sigma_x <= x[i]) & (x[i] <= voted_x + sigma_x)) {
            final_voted_x += scores[i] * x[i];
            filter_score_sum += scores[i];
        }
    }
    final_voted_x = final_voted_x / filter_score_sum;
    return final_voted_x;
}


std::vector<int> get_top_k_index(std::vector<float_t> &scores, int topk){
    if(scores.size()<topk){
        std::vector<int> index(scores.size());
        std::iota(std::begin(index), std::end(index), 0);
        return index;
    }
    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;
    for (int i=0; i<scores.size();i++){
        if(q.size()<topk)
            q.push(std::pair<double, int>(scores[i], i));
        else if(q.top().first < scores[i]){
            q.pop();
            q.push(std::pair<double, int>(scores[i], i));
        }
    }
    std::vector<int> index(topk);
    for (int i=0; i<topk;i++){
        index[topk-i-1] = q.top().second;
        q.pop();
    }
    return index;
}

void get_top_k(std::vector<float_t> &x, std::vector<int> &index){
    int n = x.size();
    for (int i=0;i < n;i++){
        if(std::find(index.begin(), index.end(), i) == index.end()){
            x[i] = 0.;
        }
    }
}

std::vector<torch::Tensor>vote_nms(torch::Tensor &bboxes,
                                   torch::Tensor &cluster_scores,
                                   torch::Tensor &vote_scores,
                                   torch::Tensor &labels,
                                   float_t nms_threshold,
                                   bool iou_enable,
                                   float_t sigma){

    auto order_indices = std::get<1>(torch::sort(cluster_scores,0,true));
    auto suppressed = torch::zeros_like(cluster_scores, torch::kBool);
    auto voted_x1 = torch::zeros_like(vote_scores, torch::kFloat32);
    auto voted_y1 = torch::zeros_like(vote_scores, torch::kFloat32);
    auto voted_x2 = torch::zeros_like(vote_scores, torch::kFloat32);
    auto voted_y2 = torch::zeros_like(vote_scores, torch::kFloat32);
    auto voted_labels = torch::zeros_like(vote_scores, torch::kInt64);
    auto voted_scores = torch::zeros_like(vote_scores, torch::kFloat32);

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
    auto cluster_scores_t = cluster_scores.data_ptr<float_t>();
    auto vote_scores_t = vote_scores.data_ptr<float_t>();
    auto suppressed_t = suppressed.data_ptr<bool>();
    auto voted_x1_t = voted_x1.data_ptr<float_t>();
    auto voted_y1_t = voted_y1.data_ptr<float_t>();
    auto voted_x2_t = voted_x2.data_ptr<float_t>();
    auto voted_y2_t = voted_y2.data_ptr<float_t>();
    auto voted_scores_t = voted_scores.data_ptr<float_t>();
    auto voted_labels_t = voted_labels.data_ptr<int64_t>();

    int ndets = cluster_scores.size(0);
    int64_t  instance_id = 0;
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
        auto vote_score_i = vote_scores_t[index];
        auto cluster_score_i = cluster_scores_t[index];
        auto area_i = (x2_i - x1_i) * (y2_i - y1_i);

        suppressed_t[index] = true;

        std::vector<int64_t> current_clustered_index;
        current_clustered_index.emplace_back(index);
        std::vector<float_t> clustered_x1, clustered_y1, clustered_x2, clustered_y2;
        std::vector<float_t> clustered_scores_from_vote, clustered_scores_from_cluster;
        clustered_x1.emplace_back(x1_i);
        clustered_y1.emplace_back(y1_i);
        clustered_x2.emplace_back(x2_i);
        clustered_y2.emplace_back(y2_i);
        clustered_scores_from_vote.emplace_back(vote_score_i);
        clustered_scores_from_cluster.emplace_back(cluster_score_i);

        for (int j=i+1; j< ndets; j++) {
            auto index_j = order_indices_t[j];
            auto label_j = labels_t[index_j];

            if (label_j != label_i) {
                continue;
            }

            if (suppressed_t[index_j]) {
                continue;
            }
            auto vote_score_j = vote_scores_t[index_j];
            auto cluster_score_j = cluster_scores_t[index_j];
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
            auto iou = inter / (area_j + area_i - inter);

            if (iou_enable){
                auto iou_factor = exp(-(1-iou) * (1-iou)/ sigma);
                vote_score_j *= iou_factor;
            }

            if (iou > nms_threshold) {
                suppressed_t[index_j] = true;
                current_clustered_index.emplace_back(index_j);
                clustered_x1.emplace_back(x1_j);
                clustered_y1.emplace_back(y1_j);
                clustered_x2.emplace_back(x2_j);
                clustered_y2.emplace_back(y2_j);
                clustered_scores_from_vote.emplace_back(vote_score_j);
                clustered_scores_from_cluster.emplace_back(cluster_score_j);

            }

        }


        auto voted_x1_i = vote_single_dim(clustered_scores_from_vote, clustered_x1);
        auto voted_y1_i = vote_single_dim(clustered_scores_from_vote, clustered_y1);
        auto voted_x2_i = vote_single_dim(clustered_scores_from_vote, clustered_x2);
        auto voted_y2_i = vote_single_dim(clustered_scores_from_vote, clustered_y2);

        voted_x1_t[instance_id] = voted_x1_i;
        voted_y1_t[instance_id] = voted_y1_i;
        voted_x2_t[instance_id] = voted_x2_i;
        voted_y2_t[instance_id] = voted_y2_i;
        voted_labels_t[instance_id] = label_i;
        voted_scores_t[instance_id] = *std::max_element(clustered_scores_from_cluster.begin(),
                                                        clustered_scores_from_cluster.end());


        instance_id ++;

    }
    auto voted_bbox = torch::stack({voted_x1, voted_y1, voted_x2, voted_y2}, -1);

    return {voted_bbox.narrow(0, 0, instance_id),
            voted_labels.narrow(0, 0, instance_id),
            voted_scores.narrow(0, 0, instance_id)};
}


std::vector<torch::Tensor>global_vote_nms(torch::Tensor &bboxes,
                                           torch::Tensor &cluster_scores,
                                           torch::Tensor &vote_scores,
                                           torch::Tensor &labels,
                                           float_t nms_threshold,
                                           bool iou_enable,
                                           float_t sigma){
    auto order_indices = std::get<1>(torch::sort(cluster_scores,0,true));
    auto suppressed = torch::zeros_like(cluster_scores, torch::kBool);
    auto voted_x1 = torch::zeros_like(vote_scores, torch::kFloat32);
    auto voted_y1 = torch::zeros_like(vote_scores, torch::kFloat32);
    auto voted_x2 = torch::zeros_like(vote_scores, torch::kFloat32);
    auto voted_y2 = torch::zeros_like(vote_scores, torch::kFloat32);
    auto voted_labels = torch::zeros_like(vote_scores, torch::kInt64);
    auto voted_scores = torch::zeros_like(vote_scores, torch::kFloat32);

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
    auto cluster_scores_t = cluster_scores.data_ptr<float_t>();
    auto vote_scores_t = vote_scores.data_ptr<float_t>();
    auto suppressed_t = suppressed.data_ptr<bool>();
    auto voted_x1_t = voted_x1.data_ptr<float_t>();
    auto voted_y1_t = voted_y1.data_ptr<float_t>();
    auto voted_x2_t = voted_x2.data_ptr<float_t>();
    auto voted_y2_t = voted_y2.data_ptr<float_t>();
    auto voted_scores_t = voted_scores.data_ptr<float_t>();
    auto voted_labels_t = voted_labels.data_ptr<int64_t>();

    std::set<int64_t> suppressed_labels;
    int ndets = cluster_scores.size(0);
    int64_t  instance_id = 0;

    for (int i=0; i<ndets; i++){
        auto index = order_indices_t[i];
        if (suppressed_t[index]){
            continue;
        }

        auto label_i = labels_t[index];
        if (!suppressed_labels.empty()){
            // if this instance's label has been finded, suppress it
            if (suppressed_labels.find(label_i) != suppressed_labels.end()){
                suppressed_t[index] = true;
                continue;
            }
        }

        auto x1_i = x1_t[index];
        auto y1_i = y1_t[index];
        auto x2_i = x2_t[index];
        auto y2_i = y2_t[index];

        auto vote_score_i = vote_scores_t[index];
        auto cluster_score_i = cluster_scores_t[index];
        auto area_i = (x2_i - x1_i) * (y2_i - y1_i);

        suppressed_t[index] = true;
        suppressed_labels.insert(label_i);

        std::vector<int64_t> current_clustered_index;
        current_clustered_index.emplace_back(index);
        std::vector<float_t> clustered_x1, clustered_y1, clustered_x2, clustered_y2;
        std::vector<float_t> clustered_scores_from_vote, clustered_scores_from_cluster;
        clustered_x1.emplace_back(x1_i);
        clustered_y1.emplace_back(y1_i);
        clustered_x2.emplace_back(x2_i);
        clustered_y2.emplace_back(y2_i);
        clustered_scores_from_vote.emplace_back(vote_score_i);
        clustered_scores_from_cluster.emplace_back(cluster_score_i);

        for (int j=i+1; j< ndets; j++) {
            auto index_j = order_indices_t[j];
            auto label_j = labels_t[index_j];

            if (label_j != label_i) {
                continue;
            }

            if (suppressed_t[index_j]) {
                continue;
            }

            auto vote_score_j = vote_scores_t[index_j];
            auto cluster_score_j = cluster_scores_t[index_j];
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
            auto iou = inter / (area_j + area_i - inter);

            if (iou_enable){
                auto iou_factor = exp(-(1-iou) * (1-iou)/ sigma);
                vote_score_j *= iou_factor;
            }

            if (iou > nms_threshold) {
                suppressed_t[index_j] = true;
                current_clustered_index.emplace_back(index_j);
                clustered_x1.emplace_back(x1_j);
                clustered_y1.emplace_back(y1_j);
                clustered_x2.emplace_back(x2_j);
                clustered_y2.emplace_back(y2_j);
                clustered_scores_from_vote.emplace_back(vote_score_j);
                clustered_scores_from_cluster.emplace_back(cluster_score_j);
            }
        }

        auto voted_x1_i = vote_single_dim(clustered_scores_from_vote, clustered_x1);
        auto voted_y1_i = vote_single_dim(clustered_scores_from_vote, clustered_y1);
        auto voted_x2_i = vote_single_dim(clustered_scores_from_vote, clustered_x2);
        auto voted_y2_i = vote_single_dim(clustered_scores_from_vote, clustered_y2);

        voted_x1_t[instance_id] = voted_x1_i;
        voted_y1_t[instance_id] = voted_y1_i;
        voted_x2_t[instance_id] = voted_x2_i;
        voted_y2_t[instance_id] = voted_y2_i;
        voted_labels_t[instance_id] = label_i;
        voted_scores_t[instance_id] = *std::max_element(clustered_scores_from_cluster.begin(),
                                                        clustered_scores_from_cluster.end());
        instance_id ++;
    }
    auto voted_bbox = torch::stack({voted_x1, voted_y1, voted_x2, voted_y2}, -1);
    return {voted_bbox.narrow(0, 0, instance_id),
            voted_labels.narrow(0, 0, instance_id),
            voted_scores.narrow(0, 0, instance_id)};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("vote_nms", &vote_nms, "vote implement for nms");
    m.def("global_vote_nms", &global_vote_nms, "global vote implement for nms");
}