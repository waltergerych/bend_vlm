import torch
import numpy as np
from kneed import KneeLocator
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import util
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

def get_cos_neighbors(query_vec, embed_dataset, k = None):
    cos_scores = util.cos_sim(query_vec.astype(float), embed_dataset['embedding'].astype(float))
    if k is None:
        _k = len(embed_dataset)
    else:
        _k = k
    top_results = torch.topk(cos_scores, k=_k)
    topk_sim = top_results.values.cpu().numpy().reshape(-1)
    top_indices = top_results.indices.cpu().numpy()[0]

    if k is None:
        kn = KneeLocator([i for i in range(len(topk_sim))], topk_sim, curve='convex', direction='decreasing').knee
        print(kn)
        top_indices = top_indices[:kn]
        topk_sim = topk_sim[:kn]
    dist_scores = 1. - topk_sim
    neighbors = embed_dataset[top_indices]
    
    return dist_scores, neighbors

def get_embeddings(input_text : list, clip_model, clip_processor, normalize=True, device='cuda'):
    # tokenized_query_text = clip_tokenizer(input_text, padding=True, return_tensors="pt").to(device)
    # with torch.no_grad():
    #     query_text_embedding = clip_model(**tokenized_query_text)['text_embeds']

    with torch.no_grad():
        inputs = clip_processor(text=input_text, return_tensors="pt", padding=True).to(device)

        query_text_embedding = clip_model.get_text_features(**inputs)#.to('cpu').numpy()

    if normalize:
        query_text_embedding /= query_text_embedding.norm(dim=-1, keepdim=True)
    return query_text_embedding

def get_proj_matrix(embeddings):
    tSVD = TruncatedSVD(n_components=len(embeddings))
    embeddings_ = tSVD.fit_transform(embeddings)
    basis = tSVD.components_.T

    # orthogonal projection
    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj = np.eye(proj.shape[0]) - proj
    return proj

def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T)

def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M  += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)
# Define the objective function with additional parameter initial_e
def objective(e_star, initial_e):
    return float(-np.dot(e_star, initial_e))

# Define the constraints with additional parameters f_mean and m_mean
def eq_dist_constraint(e_star, y_mean, x_mean):
    return float(np.dot(e_star, y_mean) - np.dot(e_star, x_mean))

def norm_constraint(e_star):
    return float(np.dot(e_star, e_star) - 1)

def legrange_text(query_embedding, ref_dataset, spurious_label, spurious_class_list, num_neighbors, proj_matrix, normalize=True):

    if proj_matrix is not None:
        query_embedding = np.matmul(query_embedding, proj_matrix.T)

    if normalize:
        norm = np.linalg.norm(query_embedding, axis=-1, keepdims=True)
        # print(norm)
        query_embedding /= norm
    
    t_embed = query_embedding.reshape(-1)
    q_t = query_embedding.reshape(1,-1)
    ref_scores, ref_samples = get_cos_neighbors(query_embedding, ref_dataset, k = len(ref_dataset))

    

    ref_embed_array = np.asarray(ref_samples['embedding'])
    ref_spurious_array = np.asarray(ref_samples[spurious_label])

    spurious_anchor_class = spurious_class_list[0]

    if num_neighbors is None:
        _sim_scores = ref_scores[ref_spurious_array == spurious_anchor_class]
        _sim_scores = 1. - _sim_scores
        kn = KneeLocator([i for i in range(len(_sim_scores))], _sim_scores, curve='convex', direction='decreasing').knee
        s_k = kn
        print(kn)
    else:
        s_k = num_neighbors

    anchor_ref_embed_array = ref_embed_array[ref_spurious_array == spurious_anchor_class][:s_k]
    # if proj_matrix is not None:
    #     anchor_ref_embed_array = np.matmul(anchor_ref_embed_array, proj_matrix.T)  
    anchor_prototype = anchor_ref_embed_array.mean(axis=0)

    #initial guess 
    x0 = query_embedding.reshape(-1).astype(float)
    # Parameters
    x_mean = anchor_prototype.reshape(-1).astype(float)

    y_means = []

    for spurious_class in spurious_class_list[1:]:
        if num_neighbors is None:
            _sim_scores = ref_scores[ref_spurious_array == spurious_class]
            _sim_scores = 1. - _sim_scores
            kn = KneeLocator([i for i in range(len(_sim_scores))], _sim_scores, curve='convex', direction='decreasing').knee
            s_k = kn
            print(kn)
        else:
            s_k = num_neighbors
        print()
        s_ref_embed_array = ref_embed_array[ref_spurious_array == spurious_class][:s_k]
        # if proj_matrix is not None:
        #     s_ref_embed_array = np.matmul(s_ref_embed_array, proj_matrix.T)       
        s_prototype = s_ref_embed_array.mean(axis=0)
        y_means.append(s_prototype)
        
    # Define the constraints in dictionary form with additional parameters
    norm_con = {'type': 'eq', 'fun': norm_constraint}
    cons = [norm_con]
    for _y_mean in y_means:
        dist_con = {'type': 'eq', 'fun': eq_dist_constraint, 'args': (_y_mean, x_mean)}
        cons.append(dist_con)

    solution = minimize(objective, x0, args=(query_embedding.reshape(-1),), method='SLSQP', constraints=cons)
    e_star = solution.x
    e_star = e_star.reshape(1,-1)

    return e_star, x_mean, y_means



def log_with_eps(x, eps=1e-10):
    if x < eps:
        return np.log(eps)
    else:
        return np.log(x)

def max_skew(returned_samples, target_dist, spurious_label='gender', target_classes = [-1,1]):
    print(target_dist)
    maxskew = 0
    for cl in target_classes:
        p_y_ds = target_dist[cl]
        p_y_returned = (np.asarray(returned_samples[spurious_label])==cl).astype(int).mean()
        # print(f"p_y_returned: {p_y_returned}, p_y_ds: {p_y_ds}")
        candidate_skew = log_with_eps(p_y_returned/p_y_ds)# np.log(p_y_returned/p_y_ds) 
        if candidate_skew > maxskew:
            maxskew = candidate_skew
    return maxskew

def get_kl(returned_samples, target_dist, spurious_label='gender', target_classes = [-1,1]):
    kl = 0
    for cl in target_classes:
        p_y_ds = target_dist[cl]
        p_y_returned = (np.asarray(returned_samples[spurious_label])==cl).astype(int).mean()
        # print(f"p_y_returned: {p_y_returned}, p_y_ds: {p_y_ds}")
        kl += p_y_returned * log_with_eps(p_y_returned/p_y_ds)#np.log(p_y_returned/p_y_ds)
    return kl

def _cl_with_max_skew(returned_samples, target_dist, spurious_label='gender', target_classes = [-1,1]):
    # print(target_dist)
    maxskew = 0
    max_skew_cl = None

    for cl in target_classes:
        p_y_ds = target_dist[cl]
        p_y_returned = (np.asarray(returned_samples[spurious_label])==cl).astype(int).mean()
        # print(f"p_y_returned: {p_y_returned}, p_y_ds: {p_y_ds}")
        candidate_skew = log_with_eps(p_y_returned/p_y_ds)#np.log(p_y_returned/p_y_ds)
        if candidate_skew > maxskew:
            maxskew = candidate_skew
            max_skew_cl = cl
    if max_skew_cl is None:
        max_skew_cl = cl
    return max_skew_cl

def _cl_with_min_skew(returned_samples, target_dist, spurious_label='gender', target_classes = [-1,1]):
    # print(target_dist)
    minskew = 100000
    min_skew_cl = None
    for cl in target_classes:
        p_y_ds = target_dist[cl]
        p_y_returned = (np.asarray(returned_samples[spurious_label])==cl).astype(int).mean()
        # print(f"p_y_returned: {p_y_returned}, p_y_ds: {p_y_ds}")
        candidate_skew = log_with_eps(p_y_returned/p_y_ds)#np.log(p_y_returned/p_y_ds)
        if candidate_skew < minskew:
            minskew = candidate_skew
            min_skew_cl = cl
    if min_skew_cl is None:
        min_skew_cl = cl
    return min_skew_cl

def group_accuracy(retrieved_samples, q_class,  spurious_label, spurious_class_list):
    s_array = np.asarray(retrieved_samples[spurious_label])
    class_array = np.asarray(retrieved_samples[q_class])
    result_dict = {}
    for s_class in spurious_class_list:
        result_dict[s_class] = (class_array[s_array == s_class] == 1).mean()
    return result_dict

def auc_roc(q_embed, embed_dataset,  q_class, spurious_label, spurious_class_list, metric_is_distance = True):
    all_scores, all_samples = get_cos_neighbors(q_embed, embed_dataset, k = len(embed_dataset))
    

    s_array = np.asarray(all_samples[spurious_label])
    print(s_array.shape)
    _class_array = np.asarray(all_samples[q_class])
    print(_class_array)
    binary_class_array = (_class_array== 1 ).astype(int)
    # print(binary_class_array)
    score_array = np.asarray(all_scores)
    if metric_is_distance:
        score_array = -1*score_array
    # print(score_array.shape)
    result_dict = {}
    for s_class in spurious_class_list:
        s_binary_class_array = binary_class_array[s_array == s_class]
        if len(np.unique(s_binary_class_array)) != 1:

            s_score_array = score_array[s_array == s_class]
            result_dict[s_class] = roc_auc_score(s_binary_class_array, s_score_array)
    return result_dict


def get_worst_group_performance(method_metric_dict, higher_better=True):
    worst_class = None
    if higher_better:
        worst_metric = 100000
    else: 
        worst_metric = -100000
    for spurious_att in method_metric_dict.keys():  
        if  higher_better:
            if method_metric_dict[spurious_att] < worst_metric:
                worst_metric = method_metric_dict[spurious_att]
                worst_class = spurious_att
        else:
            if method_metric_dict[spurious_att] > worst_metric:
                worst_metric = method_metric_dict[spurious_att]
                worst_class = spurious_att
    return worst_metric, worst_class


def get_best_group_performance(method_metric_dict, higher_better=True):
    best_class = None
    if higher_better:
        best_metric = -100000
    else: 
        best_metric = 100000
    for spurious_att in method_metric_dict.keys():  
        if  higher_better:
            if method_metric_dict[spurious_att] > best_metric:
                best_metric = method_metric_dict[spurious_att]
                best_class = spurious_att
        else:
            if method_metric_dict[spurious_att] < best_metric:
                best_metric = method_metric_dict[spurious_att]
                best_class = spurious_att
    return best_metric, best_class


def relevency(returned_samples, q_class, spurious_label='gender', spurious_class_list = [-1,1]):
    result_dict = {}
    spurious_label_array = np.asarray(returned_samples[spurious_label])
    query_class_array = np.asarray(returned_samples[q_class])
    for cl in spurious_class_list: 
        samples_for_cl = query_class_array[spurious_label_array==cl]
        p_rel = (samples_for_cl==1).astype(int).mean()
        result_dict[cl] = p_rel
    return result_dict

def get_metrics(q_embedding, query_class, att_to_debias, K, spurious_att_prior, target_spurious_class_list, target_dataset, name='Vanilla', QUERY_IS_LABELED=True):
    # result_d = {}
    # for i in range(5):
    _result = {}
    _t_ds = target_dataset
    
    _scores, _samples = get_cos_neighbors(q_embedding, _t_ds, k = K)
    # _scores, _samples = target_embeddings_dataset.get_nearest_examples(
    # "embedding", q_embedding, k=K)
    if QUERY_IS_LABELED:
        _auc_roc = auc_roc(q_embedding, _t_ds,  query_class, att_to_debias, spurious_class_list = target_spurious_class_list)
        worst_metric_val, worst_group = get_worst_group_performance(_auc_roc)
        best_metric_val, best_group = get_best_group_performance(_auc_roc)
        print(f"{name} worst group AUC ROC: {worst_metric_val}, worst group: {worst_group}")
        _result['worst_auc_roc_val'] = worst_metric_val
        _result['worst_auc_roc_group'] = worst_group

        _result['best_auc_roc_val'] = best_metric_val
        _result['best_auc_roc_group'] = best_group

        print(f"{name} gap for AUC ROC: {best_metric_val - worst_metric_val}")
        _result['auc_roc_gap'] = best_metric_val - worst_metric_val

        _relevency = relevency(_samples,  query_class, att_to_debias, spurious_class_list = target_spurious_class_list)
        worst_rel_val, worst_rel_group = get_worst_group_performance(_relevency)
        print(f"{name} worst group relevency: {worst_rel_val}, worst group: {worst_rel_group}")
        _result['worst_rel_val'] = worst_rel_val
        _result['worst_rel_group'] = worst_rel_group

    max_skew_prior = max_skew(_samples, spurious_att_prior, spurious_label = att_to_debias , target_classes=target_spurious_class_list)
    kl_prior = get_kl(_samples, spurious_att_prior, spurious_label = att_to_debias , target_classes=target_spurious_class_list)
    print(f"{name} Max Skew Prior: {max_skew_prior}")
    print(f"{name} KL Prior: {kl_prior}")
    
    

    _result['max_skew_prior'] = max_skew_prior
    _result['kl_prior'] = kl_prior
    result_d = _result
    # result_d[f"fold_{i}"] = _result
    print()
    return result_d