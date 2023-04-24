#include <torch/extension.h>
#include <vector>
#include <map>

using namespace torch::indexing;

torch::Tensor FastMBD(torch::Tensor &image , torch::Tensor &seeds_x, torch::Tensor &seeds_y, float alpha, int niter, int base_size){
//    image format: H*W*C
    int image_h, image_w;
    image_h = image.size(0);
    image_w = image.size(1);
    double size_factor = 0;
    if (image_h * image_w < base_size * base_size){
        size_factor = 400.;
    }else{
        size_factor = 400. * (image_w * image_h / (base_size * base_size));
    }
    double factor = alpha * alpha / size_factor;

    torch::Tensor labelmap = torch::full({image_h, image_w}, -1, image.options().dtype(torch::kInt32));
    torch::Tensor dmap = torch::full({image_h, image_w}, 255, image.options().dtype(torch::kFloat64));


    auto seed_y_t = seeds_y.data_ptr<int64_t>();
    auto seed_x_t = seeds_x.data_ptr<int64_t>();

    labelmap.index_put_({seeds_y, seeds_x}, torch::arange(0, seeds_y.size(0), 1, labelmap.options()));
    dmap.index_put_({seeds_y, seeds_x}, 0);
    auto H =  image.clone(), L = image.clone();


    auto H_t = H.data_ptr<uint8_t>();
    auto L_t = L.data_ptr<uint8_t>();

    int start_x, end_x, start_y, end_y;
    std::vector<int> offset_x(2,0);
    std::vector<int> offset_y(2,0);
    int step;

    uint8_t *current_point;
    uint8_t *h;
    uint8_t *l;

    int neighbor_point_x, neighbor_point_y;

    auto label_t = labelmap.data_ptr<int32_t>();

    for (int i=0; i<niter; i++) {

        if (i % 2 == 0) {
            start_x = 0;
            end_x = image_w;
            start_y = 0;
            end_y = image_h;
            offset_x[0] = 0;
            offset_x[1] = -1;
            offset_y[0] = -1;
            offset_y[1] = 0;
            step = 1;
        } else {
            start_x = image_w - 1;
            end_x = -1;
            start_y = image_h - 1;
            end_y = -1;
            offset_x[0] = 0;
            offset_x[1] = 1;
            offset_y[0] = 1;
            offset_y[1] = 0;
            step = -1;
        }


        for (int y = start_y; y != end_y; y += step) {
            auto current_row = image[y].data_ptr<uint8_t>();
            auto current_h_row = H[y].data_ptr<uint8_t>();
            auto current_l_row = L[y].data_ptr<uint8_t>();
            auto label_t_ = labelmap[y].data_ptr<int32_t>();
            auto dmap_t = dmap[y].data_ptr<double_t>();

            for (int x = start_x; x != end_x; x += step) {

                current_point = current_row+x*3;
                for (auto &k:{0, 1}) {
                    neighbor_point_x = x + offset_x[k];
                    neighbor_point_y = y + offset_y[k];
                    if (neighbor_point_x >=0 && neighbor_point_x < image_w && neighbor_point_y >=0 && neighbor_point_y < image_h){
                        auto neighbor_point_label = label_t[neighbor_point_y*image_w+neighbor_point_x];
                        if (neighbor_point_label >=0 ){
                            h = H_t + (neighbor_point_y*image_w + neighbor_point_x)*3;
                            l = L_t + (neighbor_point_y*image_w + neighbor_point_x)*3;

                            uint8_t max_cost[3], min_cost[3];
                            int cost_channel[3];
                            for (int c = 0; c < 3; c++) {
                                max_cost[c] = std::max(*(h+c), *(current_point+c));
                                min_cost[c] = std::min(*(l+c), *(current_point+c));
                                cost_channel[c] = max_cost[c] - min_cost[c];
                            }

                            double cost=0;
                            cost += std::max(std::max(cost_channel[0], cost_channel[1]), cost_channel[2]) / 255.;
                            cost *= cost;

                            auto seed_y = seed_y_t[neighbor_point_label];
                            auto seed_x = seed_x_t[neighbor_point_label];

                            cost += factor * ((seed_y - y)*(seed_y - y) + (seed_x -x)*(seed_x - x));

                            if (cost < dmap_t[x]){
                                dmap_t[x] = cost;
                                label_t_[x] = neighbor_point_label;
                                memcpy(current_h_row+x*3, max_cost, 3*sizeof(uint8_t));
                                memcpy(current_l_row+x*3, min_cost, 3*sizeof(uint8_t));
                            }
                        }

                    }
                }
            }
        }
    }

    return dmap;
}



torch::Tensor MBD(torch::Tensor &image, torch::Tensor &seeds_x, torch::Tensor &seeds_y,
                    float alpha, int niter, int base_size){

    auto dmap = FastMBD(image, seeds_x, seeds_y, alpha, niter, base_size);
    return dmap;
}

torch::Tensor GeodesicDistanceTransform(torch::Tensor &cost, torch::Tensor &distance_map, torch::Tensor &label_map, int w, int h)
{
    float c1 = 1.0f / 2.0f;
    float c2 = sqrt(2.0f) / 2.0f;
    float d = 0.0f;
    int i, j;
    float *dist_row, *cost_row;
    float *dist_row_prev, *cost_row_prev;
    int32_t *label_row;
    int32_t *label_row_prev;
    auto cost_t = cost.data_ptr<float>();
    auto distance_t = distance_map.data_ptr<float>();
    auto labels_t = label_map.data_ptr<int32_t>();



#define UPDATE(cur_dist,cur_label,cur_cost,prev_dist,prev_label,prev_cost,coef)\
		{\
    d = prev_dist + coef*(cur_cost+prev_cost);\
    if(cur_dist>d){\
        cur_dist=d;\
        cur_label = prev_label;}\
		}

    //first pass (left-to-right, top-to-bottom):
    dist_row = distance_t;
    label_row = labels_t;
    cost_row = cost_t;
    for (j = 1; j < w; j++)
    UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);

    for (i = 1; i < h; i++)
    {
        dist_row = distance_t + i*w;
        dist_row_prev = distance_t + (i - 1) * w;

        label_row = labels_t + i * w;
        label_row_prev = labels_t + (i - 1)*w;

        cost_row = cost_t + i * w;
        cost_row_prev = cost_t + (i - 1) * w;

        j = 0;
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
        j++;
        for (; j < w - 1; j++)
        {
            UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
            UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
            UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
            UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
        }
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
    }

    //second pass (right-to-left, bottom-to-top):
    dist_row = distance_t + (h - 1) * w;
    label_row = labels_t + (h - 1)*w;
    cost_row = cost_t + (h - 1)*w;
    for (j = w - 2; j >= 0; j--)
    UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);

    for (i = h - 2; i >= 0; i--)
    {
        dist_row = distance_t + i*w;
        dist_row_prev = distance_t + (i + 1) * w;

        label_row = labels_t + i * w;
        label_row_prev = labels_t + (i + 1)*w;

        cost_row = cost_t + i * w;
        cost_row_prev = cost_t + (i + 1) * w;

        j = w - 1;
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
        j--;
        for (; j > 0; j--)
        {
            UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);
            UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
            UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
            UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
        }
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
        UPDATE(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
    }
#undef CHECK
    return distance_map;
}

torch::Tensor GDT(torch::Tensor &costmap, torch::Tensor &seeds_x, torch::Tensor &seeds_y){
    int image_h, image_w;
    image_h = costmap.size(0);
    image_w = costmap.size(1);
    auto label_map = torch::full({image_h, image_w}, -1, torch::kInt32);
    auto distance_map = torch::full({image_h, image_w}, 255, torch::kFloat32);
    label_map.index_put_({seeds_y, seeds_x}, torch::arange(0, seeds_y.size(0), 1, label_map.options()));
    distance_map.index_put_({seeds_y, seeds_x}, costmap.index({seeds_y, seeds_x}));
    distance_map = GeodesicDistanceTransform(costmap, distance_map, label_map, image_w, image_h);
    return distance_map;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("MBD", &MBD, "Superpixel segmentation by minimum barrier distance transform");
    m.def("GDT", &GDT, "Superpixel segmentation by Geodesic distance transform");
}