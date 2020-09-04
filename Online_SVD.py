import numpy as np
import sparsesvd
import scipy.sparse as sps

# This function follow the Algorithm of Appendix A of Paper 'Fast online SVD revisions for lightweight recommender systems'
# update the svd from X to X + A @ B.T
# Param is U, S, V of X and A, B
# Return new U, S, V
def increment_svd(U, S, V, A, B):
    # print(U)
    # print(S)
    # print(V)
    # print(A)
    # print(B)
    u_x, u_y = U.shape 
    rank_u = u_y 
    a_x, a_y = A.shape 

    ua = np.concatenate((U, A), axis=1) 
    # print(ua)
    qp,rp = np.linalg.qr(ua, mode='complete') 

    qp_x, qp_y = qp.shape 
    rp_x, rp_y = rp.shape 
    # print(qp)
    # print(rp)
    # print("ua shape {}, qp shape{}, rp shape{}, rank_u {}".format(ua.shape, qp.shape, rp.shape,rank_u))

    if qp_y == rank_u:
        P = qp[:,rank_u : qp_y]
        Ra = rp[rank_u:rp_x , rank_u:rp_y].T
        dim_Ra = 0
    else:
        P = qp[:,rank_u : qp_y]
        Ra = rp[rank_u:rp_x , rank_u:rp_y].T
        dim_Ra = rp_x - rank_u

    # print(P)
    # print(Ra)
    
    v = rank_u + a_y
    # M = rp[0:rank_u, rank_u:v]

    ub = np.concatenate((V, B), axis=1)
    # print("\n")
    # print(ub)
    qq, rq = np.linalg.qr(ub, mode='complete')
    qq_x,qq_y = qq.shape
    rq_x, rq_y = rq.shape

    # print(qq)
    # print(rq)
    # print("ub shape {}, qq shape{}, rq shape{}".format(ub.shape, qq.shape, rq.shape))


    if qq_y == rank_u:
        Q = qq[:,rank_u : qq_y]
        Rb = rq[rank_u:rq_x , rank_u:rq_y]
        dim_Rb = 0
    else:
        Q = qq[:,rank_u : qq_y]
        Rb = rq[rank_u:rq_x , rank_u:rq_y]
        dim_Rb = rq_x - rank_u

    # print(Q)
    # print(Rb)
    M = U.T @ A
    N = V.T @ B
    # print(N)

    ##########################################
    ####### Creating Augmented Sigma
    ##########################################
    Ra_flatten = Ra.T
    temp_0 = np.concatenate((M, Ra_flatten), axis=0)
    # print(temp_0)

    if Rb.shape[1] == 0:
        temp_1 = N
    else:
        temp_1 = np.concatenate((N, Rb), axis=0)
    # print(temp_1)
    
    # print('\ntemp0 shape ', temp_0.shape)
    # print('\ntemp1 shape ', temp_1.shape)
    S_augmented = temp_0 @ temp_1.T
    # print(S_augmented)

    S_new = np.zeros((rank_u + dim_Ra, rank_u + dim_Rb))
    S_new[0:S.shape[0], 0: S.shape[1]] = S
    S_new = S_new + S_augmented
    # print(S_new)

    # u, sig, vh = np.linalg.svd(S_new, full_matrices=False)
    temp = sps.csc_matrix(S_new)
    u, sig, vh = sparsesvd.sparsesvd(temp, min(S_new.shape))
    # print("u shape {}".format(u.shape))
    u = u.T


    U_new = np.concatenate((U,P),axis=1) @ u
    S_new = np.diag(sig)

    V_new = np.concatenate((V,Q),axis=1) @ vh.T

    return U_new, S_new, V_new

# This function update the svd when X is appended with more lines
def append_lines_update_svd(m_old, m_new):
    if m_old.shape[1] != m_new.shape[1]:
        print('\nAppend_Lines:New matrix muss has the same colums as old matrix!')
        return
    m,n = m_old.shape
    factor = min(m,n)
    
    # factor = 5
    c = m_new.shape[0] - m
    # print("c {}, factor {}".format(c, factor))
    temp = sps.csc_matrix(m_old)
    u, s, v = sparsesvd.sparsesvd(temp, factor)
  
    u = u.T
    # print("u shape {}".format(u.shape))


    B = m_new[m:, :].T
    A = np.concatenate((np.zeros((m, c)),np.eye(c)), axis=0)
    S = np.diag(s)
    U = np.concatenate((u,np.zeros((c,u.shape[1]))),axis=0)
    V = v.T
    # print("u shape {}, A shape {}, U shape {}, V shape {}".format(u.shape, A.shape, U.shape, V.shape))
    u_new, s_new, v_new = increment_svd(U, S, V , A, B)

    e = np.linalg.norm(m_new - np.dot(u_new , np.dot( s_new, v_new.T)),2)
    print('Error is', e)

###############
# This function update svd when single row or colum in X is updated
# not necessary
###############
def single_line_update_svd(m_old, m_new):
    if m_old.shape != m_new.shape:
        print("\nSingel_line: Matrix shape don't match")
    shape = m_old.shape
    diff = m_new - m_old
    r,c = np.nonzero(diff)
    # print(r,c)
    x = diff[diff != 0]
    # print(x, x.shape)
    if x.shape[0] == shape[0]:
        # update colum
        col = c[0]
        a = x.reshape((shape[0],1))
        b = np.zeros((shape[1],1))
        b[c,0] = 1
    else:
        # update row
        row = r[0]
        a = np.zeros((shape[0],1))
        a[row,0] = 1
        b = x.reshape((shape[1],1)) 
    # print(np.allclose(m_new, m_old + a@b.T))
    temp = sps.csc_matrix(m_old)
    u, s, v = sparsesvd.sparsesvd(temp, min(shape))
    u = u.T 
    v = v.T 
    s = np.diag(s)
    u_new, s_new, v_new = increment_svd(u,s,v,a,b)
    e = np.linalg.norm(m_new - (u_new @ s_new @ v_new.T), 2)
    print("Error is ", e)

def multi_lines_update_svd(m_old, m_new):
    if m_old.shape != m_new.shape:
        print("\Multi_lines: Matrix shape don't match")
    shape = m_old.shape
    diff = m_new - m_old
    r,c = np.nonzero(diff)
    # get the indexs of the rows and colums which are updated
    rows = np.unique(r)
    cols = np.unique(c)
    # print("rows are ", rows)
    if len(rows) <= len(cols):
        print("row update")
        a = np.zeros((shape[0],len(rows)))
        b = np.zeros((shape[1],len(rows)))
        for idx, i in enumerate(rows):
            a[i,int(idx)] = 1
            b[:,int(idx)] = diff[i,:].T
        # print(m_new)
        # print(a @ b.T)
    else:
        print("col update")
        a = np.zeros((shape[0],len(cols)))
        b = np.zeros((shape[1],len(cols)))
        for idx, i in enumerate(cols):
            a[:, int(idx)] = diff[:,i]
            b[i, int(idx)] = 1
        # print(m_new)
        # print(a @ b.T)
    temp = sps.csc_matrix(m_old)
    u, s, v = sparsesvd.sparsesvd(temp, min(shape))
    u = u.T 
    v = v.T 
    s = np.diag(s)
    u_new, s_new, v_new = increment_svd(u,s,v,a,b)
    e = np.linalg.norm(m_new - (u_new @ s_new @ v_new.T), 2)
    print("Error is ", e)


if __name__ == '__main__':
    m = 50
    n = 40
    X = np.random.uniform(0,5, (m,n))
    # X= np.ones((m,n))
    # X[3,4] = np.random.randint(0,10)
   
    # ###############
    # # append rows
    # ###############
    # new_data = np.random.uniform(0,4,(2,n))
    # m_new = np.concatenate((X,new_data), axis=0)
    # # append_lines_update_svd(X, m_new)
 
    # # ###############
    # # update single col
    # ###############
    # update = np.random.uniform(0,10,(m,))
    # temp = np.random.randint(0,n-1)
    # X_new = np.copy(X)
    # X_new[:, temp] = update
    # # single_line_update_svd(X, X_new)
    # multi_lines_update_svd(X,X_new)

    # ###############
    # # update single row
    # ###############
    # update = np.random.uniform(0,10,(n,))
    # temp = np.random.randint(0,m-1)
    # X_new = np.copy(X)
    # X_new[temp, :] = update
    # #single_line_update_svd(X, X_new)
    # multi_lines_update_svd(X,X_new)

    # ###############
    # # update rows
    # ###############
    # new = np.copy(X)
    # random_rows = np.random.permutation(m)[:2]
    # # print(random_rows)
    # for i in random_rows:
    #     up = np.random.uniform(0,10,(n,))
    #     new[i, :] = up
    # # print(new)
    # multi_lines_update_svd(X, new)

    # ###############
    # # update cols
    # ###############
    # new = np.copy(X)
    # random_cols = np.random.permutation(n)[:1]
    # # print(random_rows)
    # for i in random_cols:
    #     up = np.random.uniform(0,10,(m,))
    #     new[:, i] = up
    # # print(new)
    # multi_lines_update_svd(X, new)

    ###############
    # update random positions
    ###############
    new = np.copy(X)
    random_rows = np.random.randint(0,m,size=20)
    random_cols = np.random.randint(0,n,size=10)
    # print(random_rows,random_cols)
    for i in random_rows:
        for j in random_cols:
            new[i][j] = np.random.uniform(0,10)
    # print(new)
    multi_lines_update_svd(X, new)

    

   

    











