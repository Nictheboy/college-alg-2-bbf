#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath>   // For std::sqrt
#include <limits>  // For std::numeric_limits
#include <memory>  // For std::unique_ptr

// --- Point Structure (same as before) ---
struct Point
{
    std::vector<int> coordinates;
    int id;

    Point() : id(-1) {}
    Point(int k) : coordinates(k), id(-1) {}
    Point(const std::vector<int> &coords, int _id = -1) : coordinates(coords), id(_id) {}

    bool operator==(const Point &other) const
    {
        return coordinates == other.coordinates;
    }
};

// --- Distance Function (Squared Euclidean - same as before) ---
long long distance_sq(const Point &p1, const Point &p2)
{
    if (p1.coordinates.empty() || p2.coordinates.empty() || p1.coordinates.size() != p2.coordinates.size())
    {
        return std::numeric_limits<long long>::max();
    }
    long long dist = 0;
    for (size_t i = 0; i < p1.coordinates.size(); ++i)
    {
        long long diff = static_cast<long long>(p1.coordinates[i]) - p2.coordinates[i];
        dist += diff * diff;
    }
    return dist;
}

// --- KD-Tree Node and Class (minor changes for encapsulation, largely same) ---
namespace KDTreeLib
{ // Encapsulate KD-Tree implementation details
    struct KDNode
    {
        Point point;
        KDNode *left = nullptr;
        KDNode *right = nullptr;
        int split_dim = -1;

        KDNode(const Point &p) : point(p) {}
    };

    class KDTree
    {
    public:
        KDNode *root = nullptr;
        int k_dim;
        size_t nodes_created = 0; // For potential internal use, not directly for space output

        KDTree(int k) : k_dim(k) {}

        ~KDTree()
        {
            destroy_recursive(root);
        }

        void build(const std::vector<Point> &points)
        {
            if (points.empty())
                return;
            std::vector<Point> modifiable_points = points;
            nodes_created = 0;
            root = build_recursive(modifiable_points, 0, modifiable_points.size(), 0);
        }

        Point search_exact(const Point &query)
        {
            if (!root)
                return Point();
            Point best_guess = root->point;
            long long min_dist_sq_val = distance_sq(query, root->point);
            search_exact_recursive(root, query, 0, best_guess, min_dist_sq_val);
            return best_guess;
        }

        Point search_approx(const Point &query, int max_attempts, int &attempts_made_out)
        {
            if (!root)
                return Point();
            Point best_guess = root->point;
            long long min_dist_sq_val = distance_sq(query, root->point);
            int attempts_count = 1;

            search_approx_recursive(root, query, 0, best_guess, min_dist_sq_val, max_attempts, attempts_count);
            attempts_made_out = attempts_count;
            return best_guess;
        }

    private:
        KDNode *build_recursive(std::vector<Point> &points, size_t start, size_t end, int depth)
        {
            if (start >= end)
                return nullptr;
            int axis = depth % k_dim;
            size_t median_idx = start + (end - start) / 2;
            std::nth_element(points.begin() + start, points.begin() + median_idx, points.begin() + end,
                             [&](const Point &a, const Point &b)
                             {
                                 return a.coordinates[axis] < b.coordinates[axis];
                             });
            KDNode *node = new KDNode(points[median_idx]);
            node->split_dim = axis;
            nodes_created++;
            node->left = build_recursive(points, start, median_idx, depth + 1);
            node->right = build_recursive(points, median_idx + 1, end, depth + 1);
            return node;
        }

        void search_exact_recursive(KDNode *current, const Point &query, int depth, Point &best_point, long long &min_dist_sq_ref)
        {
            if (!current)
                return;
            long long d_sq = distance_sq(query, current->point);
            if (d_sq < min_dist_sq_ref)
            {
                min_dist_sq_ref = d_sq;
                best_point = current->point;
            }
            int axis = current->split_dim;
            long long axis_dist_sq = static_cast<long long>(query.coordinates[axis] - current->point.coordinates[axis]) *
                                     (query.coordinates[axis] - current->point.coordinates[axis]);
            KDNode *first_child = (query.coordinates[axis] < current->point.coordinates[axis]) ? current->left : current->right;
            KDNode *second_child = (query.coordinates[axis] < current->point.coordinates[axis]) ? current->right : current->left;
            search_exact_recursive(first_child, query, depth + 1, best_point, min_dist_sq_ref);
            if (axis_dist_sq < min_dist_sq_ref)
            {
                search_exact_recursive(second_child, query, depth + 1, best_point, min_dist_sq_ref);
            }
        }

        void search_approx_recursive(KDNode *current, const Point &query, int depth, Point &best_point, long long &min_dist_sq_ref, int max_attempts, int &attempts_count)
        {
            if (!current || attempts_count >= max_attempts)
                return;
            if (current != root)
            {
                attempts_count++;
                if (attempts_count > max_attempts)
                    return;
            }
            long long d_sq = distance_sq(query, current->point);
            if (d_sq < min_dist_sq_ref)
            {
                min_dist_sq_ref = d_sq;
                best_point = current->point;
            }
            if (attempts_count >= max_attempts)
                return;
            int axis = current->split_dim;
            long long axis_dist_sq = static_cast<long long>(query.coordinates[axis] - current->point.coordinates[axis]) *
                                     (query.coordinates[axis] - current->point.coordinates[axis]);
            KDNode *first_child = (query.coordinates[axis] < current->point.coordinates[axis]) ? current->left : current->right;
            KDNode *second_child = (query.coordinates[axis] < current->point.coordinates[axis]) ? current->right : current->left;
            search_approx_recursive(first_child, query, depth + 1, best_point, min_dist_sq_ref, max_attempts, attempts_count);
            if (attempts_count >= max_attempts)
                return;
            if (axis_dist_sq < min_dist_sq_ref)
            {
                search_approx_recursive(second_child, query, depth + 1, best_point, min_dist_sq_ref, max_attempts, attempts_count);
            }
        }

        void destroy_recursive(KDNode *node)
        {
            if (node)
            {
                destroy_recursive(node->left);
                destroy_recursive(node->right);
                delete node;
            }
        }
    };
} // namespace KDTreeLib

// --- Abstract Search Algorithm ---
class SearchAlgorithm
{
public:
    virtual ~SearchAlgorithm() = default;
    virtual std::string get_name() const = 0;

    virtual void build(const std::vector<Point> &data_points) = 0;
    virtual Point find_nearest(const Point &query_point) = 0; // Renamed from search to avoid conflict

    // For calculating overall accuracy for approximate methods
    // Takes the query points, the results found by *this* algorithm, and the ground truth results
    virtual double calculate_accuracy_metric(
        const std::vector<Point> &queries,
        const std::vector<Point> &algorithm_results,
        const std::vector<Point> &ground_truth_results) const
    {
        // Default for exact algorithms or those not needing this specific ratio metric
        return 100.0; // Or NaN, or throw error if not applicable
    }
};

// --- Brute Force Algorithm ---
class BruteForceAlgorithm : public SearchAlgorithm
{
private:
    std::vector<Point> points_to_search;
    int k_dim_;

public:
    BruteForceAlgorithm(int k_dim) : k_dim_(k_dim) {}

    std::string get_name() const override { return "BruteForce"; }

    void build(const std::vector<Point> &data_points) override
    {
        points_to_search = data_points;
    }

    Point find_nearest(const Point &query_point) override
    {
        if (points_to_search.empty())
        {
            return Point(k_dim_); // Return an invalid/empty point
        }
        Point best_point = points_to_search[0];
        long long min_dist_sq = distance_sq(query_point, points_to_search[0]);

        for (size_t i = 1; i < points_to_search.size(); ++i)
        {
            long long current_dist_sq = distance_sq(query_point, points_to_search[i]);
            if (current_dist_sq < min_dist_sq)
            {
                min_dist_sq = current_dist_sq;
                best_point = points_to_search[i];
            }
        }
        return best_point;
    }
    // Accuracy for BF is implicitly 100% as it's the ground truth.
    // The default calculate_accuracy_metric returns 100.0, which is fine.
};

// --- Base KD-Tree Algorithm (to share KD-Tree instance) ---
class KDTreeAlgorithmBase : public SearchAlgorithm
{
protected:
    KDTreeLib::KDTree kd_tree_instance;
    int k_dim_;

public:
    KDTreeAlgorithmBase(int k_dim) : kd_tree_instance(k_dim), k_dim_(k_dim) {}

    void build(const std::vector<Point> &data_points) override
    {
        kd_tree_instance.build(data_points);
    }
};

// --- Standard KD-Tree Algorithm ---
class KDTreeExactAlgorithm : public KDTreeAlgorithmBase
{
public:
    KDTreeExactAlgorithm(int k_dim) : KDTreeAlgorithmBase(k_dim) {}

    std::string get_name() const override { return "KDTree_Exact"; }

    Point find_nearest(const Point &query_point) override
    {
        return kd_tree_instance.search_exact(query_point);
    }
    // Accuracy for exact KD-Tree against BF should be 100%.
    // We can add a check here for verification if needed, but for reporting, 100 is assumed.
    double calculate_accuracy_metric(
        const std::vector<Point> &queries,
        const std::vector<Point> &algorithm_results,
        const std::vector<Point> &ground_truth_results) const override
    {
        if (queries.empty())
            return 100.0; // No queries, perfect accuracy.
        int correct_matches = 0;
        for (size_t i = 0; i < algorithm_results.size(); ++i)
        {
            // Exact match means distance between alg_result and ground_truth_result is 0
            if (distance_sq(algorithm_results[i], ground_truth_results[i]) == 0)
            {
                correct_matches++;
            }
        }
        return (static_cast<double>(correct_matches) / algorithm_results.size()) * 100.0;
    }
};

// --- Approximate KD-Tree Algorithm (BBF-like) ---
class KDTreeApproximateAlgorithm : public KDTreeAlgorithmBase
{
private:
    int max_attempts_;

public:
    KDTreeApproximateAlgorithm(int k_dim, int max_attempts)
        : KDTreeAlgorithmBase(k_dim), max_attempts_(max_attempts) {}

    std::string get_name() const override { return "KDTree_Approx_BBF" + std::to_string(max_attempts_); }

    Point find_nearest(const Point &query_point) override
    {
        int attempts_made = 0; // Not strictly used by caller here, but good for debugging
        return kd_tree_instance.search_approx(query_point, max_attempts_, attempts_made);
    }

    double calculate_accuracy_metric(
        const std::vector<Point> &queries,
        const std::vector<Point> &algorithm_results,
        const std::vector<Point> &ground_truth_results) const override
    {
        if (queries.empty())
        {
            return 100.0; // Or NaN if M=0 is an issue for "percentage"
        }

        int accurate_queries_count = 0;
        for (size_t i = 0; i < queries.size(); ++i)
        {
            const Point &query_pt = queries[i];
            const Point &approx_result = algorithm_results[i];
            const Point &exact_result = ground_truth_results[i];

            long long dist_sq_approx = distance_sq(query_pt, approx_result);
            long long dist_sq_exact = distance_sq(query_pt, exact_result);

            if (dist_sq_exact == 0)
            { // Query point matches a point in dataset
                if (dist_sq_approx == 0)
                { // Approx also found the exact match
                    accurate_queries_count++;
                }
                // Else: exact is 0, approx is >0. Ratio is inf. Not accurate by 1.05 rule.
            }
            else
            {
                double ratio = std::sqrt(static_cast<double>(dist_sq_approx)) / std::sqrt(static_cast<double>(dist_sq_exact));
                if (ratio <= 1.05)
                {
                    accurate_queries_count++;
                }
            }
        }
        return (static_cast<double>(accurate_queries_count) / queries.size()) * 100.0;
    }
};

int main()
{
    std::ofstream results_file("results.txt");
    // New header: dataset_id,algorithm_name,build_time_ms,query_time_ms,accuracy_percent
    results_file << "dataset_id,algorithm_name,build_time_ms,query_time_ms,accuracy_percent\n";
    results_file << std::fixed << std::setprecision(6);

    const int MAX_BBF_ATTEMPTS = 200;

    for (int i = 1; i <= 100; ++i)
    {
        std::string filename = "data/" + std::to_string(i) + ".txt";
        std::ifstream infile(filename);
        if (!infile.is_open())
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            continue;
        }

        int N, M, K_dim;
        infile >> N >> M >> K_dim;

        std::vector<Point> build_points(N);
        for (int j = 0; j < N; ++j)
        {
            build_points[j].id = j; // Assign an ID
            build_points[j].coordinates.resize(K_dim);
            for (int k_idx = 0; k_idx < K_dim; ++k_idx)
            {
                infile >> build_points[j].coordinates[k_idx];
            }
        }

        std::vector<Point> query_points(M);
        for (int j = 0; j < M; ++j)
        {
            query_points[j].coordinates.resize(K_dim);
            for (int k_idx = 0; k_idx < K_dim; ++k_idx)
            {
                infile >> query_points[j].coordinates[k_idx];
            }
        }
        infile.close();

        std::cout << "Processing dataset " << i << " (N=" << N << ", M=" << M << ", K=" << K_dim << ")" << std::endl;

        // --- Ground Truth Generation (Brute Force) ---
        BruteForceAlgorithm bf_alg(K_dim);
        double bf_build_time_ms = 0;
        double bf_query_time_ms = 0;
        std::vector<Point> ground_truth_results;
        ground_truth_results.reserve(M);

        if (N > 0)
        {
            auto build_start_time = std::chrono::high_resolution_clock::now();
            bf_alg.build(build_points);
            auto build_end_time = std::chrono::high_resolution_clock::now();
            bf_build_time_ms = std::chrono::duration<double, std::milli>(build_end_time - build_start_time).count();

            if (M > 0)
            {
                auto query_start_time = std::chrono::high_resolution_clock::now();
                for (const auto &q_pt : query_points)
                {
                    ground_truth_results.push_back(bf_alg.find_nearest(q_pt));
                }
                auto query_end_time = std::chrono::high_resolution_clock::now();
                bf_query_time_ms = std::chrono::duration<double, std::milli>(query_end_time - query_start_time).count();
            }
        }
        results_file << i << ","
                     << bf_alg.get_name() << ","
                     << bf_build_time_ms << ","
                     << bf_query_time_ms << ","
                     << "100.0" // Brute force is the ground truth, so 100% accurate by definition
                     << "\n";
        std::cout << "  " << bf_alg.get_name() << ": Build=" << bf_build_time_ms << "ms, Query=" << bf_query_time_ms << "ms, Acc=100.0%" << std::endl;

        // --- List of Algorithms to Test (excluding BruteForce as it's already run) ---
        std::vector<std::unique_ptr<SearchAlgorithm>> algorithms_to_test;
        algorithms_to_test.emplace_back(std::make_unique<KDTreeExactAlgorithm>(K_dim));
        algorithms_to_test.emplace_back(std::make_unique<KDTreeApproximateAlgorithm>(K_dim, MAX_BBF_ATTEMPTS));
        // Add more algorithms here if needed

        for (auto &alg_ptr : algorithms_to_test)
        {
            SearchAlgorithm &alg = *alg_ptr; // Dereference unique_ptr
            double current_alg_build_time_ms = 0;
            double current_alg_query_time_ms = 0;
            std::vector<Point> current_alg_results;
            current_alg_results.reserve(M);

            if (N > 0)
            {
                auto build_start_time = std::chrono::high_resolution_clock::now();
                alg.build(build_points);
                auto build_end_time = std::chrono::high_resolution_clock::now();
                current_alg_build_time_ms = std::chrono::duration<double, std::milli>(build_end_time - build_start_time).count();

                if (M > 0)
                {
                    auto query_start_time = std::chrono::high_resolution_clock::now();
                    for (const auto &q_pt : query_points)
                    {
                        current_alg_results.push_back(alg.find_nearest(q_pt));
                    }
                    auto query_end_time = std::chrono::high_resolution_clock::now();
                    current_alg_query_time_ms = std::chrono::duration<double, std::milli>(query_end_time - query_start_time).count();
                }
            }

            double accuracy_metric = 0.0;
            if (M > 0 && N > 0)
            { // Accuracy calculation requires queries and ground truth
                accuracy_metric = alg.calculate_accuracy_metric(query_points, current_alg_results, ground_truth_results);
            }
            else if (M == 0 || N == 0)
            {                            // If no queries or no data to build on
                accuracy_metric = 100.0; // Or some other default, like NaN if Python handles it better
            }

            results_file << i << ","
                         << alg.get_name() << ","
                         << current_alg_build_time_ms << ","
                         << current_alg_query_time_ms << ","
                         << accuracy_metric
                         << "\n";
            std::cout << "  " << alg.get_name() << ": Build=" << current_alg_build_time_ms << "ms, Query=" << current_alg_query_time_ms << "ms, Acc=" << accuracy_metric << "%" << std::endl;
        }
    }

    results_file.close();
    std::cout << "Processing complete. Results saved to results.txt" << std::endl;
    return 0;
}
