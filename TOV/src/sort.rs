use ndarray::{Array3, Axis, ArrayView2};

const M_SORT: f64 = 1.4;

fn find_l(array: ArrayView2<f64>) -> Option<f64> {
    /* 
    find lambda value at mass M_SORT
    input is Array2 of shape (4, n), where the 1st row is mass, 2nd row is lambda
    output is Option(lambda at M_SORT)
    */
    let index = array.row(1).iter().position(|x| x>= &M_SORT);

    match index{
        Some(index) => {
        let l = array.row(2).get(index).unwrap().clone();
        return Some(l);
        }
        None => return None
    }
}

fn find_m(array: ArrayView2<f64>) -> f64 {
    /* 
    find highest mass that does not reach M_SORT
    input is 2d array of shape (4, n), where the 1st row is mass
    output is Option(mass)
    */
    let binding = array.row(1);
    let m= binding
                                .iter()
                                .rfind(|x| *x<= &M_SORT)
                                .expect("Could not find mass for sorting");
    
    m.clone()
}

pub fn sort_indices(array: &Array3<f64>) -> Vec<usize>{
    /*
    Sorting based on ascending order of lambda at M_SORT
    Input is a Array3 of shape (4, m, n) 
    where the first Axis is radius, mass, lambda, p_c
    the second axis is m number of EOS sets,
    and third axis is n number of points in the MR/Tidal curves
    Returns the indices in ascending order of lambda at M_SORT. 
    Indices should be used to sort second axis.
    */
    let mut m = Vec::new();
    let mut m_i = Vec::new();
    let mut l = Vec::new();
    let mut l_i= Vec::new();
    
    for (i, results_slice) in array.axis_iter(Axis(1)).enumerate() {
        let l_temp = find_l(results_slice); 

        match l_temp {
            Some(l_temp) => {
                l.push(l_temp);
                l_i.push(i);
            },
            None => {
                let m_temp = find_m(results_slice); // if MR curve doesn't reach M_SORT, sort by mass instead
                  m.push(m_temp);
                  m_i.push(i);
            }
        }
    }

    let mut zipped_l: Vec<(&f64, &usize)> = l.iter()
                        .zip(l_i.iter())
                        .collect();
    zipped_l.sort_by(|a, b| a.0.partial_cmp(b.0).expect("Could not sort using lambda_1.4"));


    let mut zipped_m: Vec<(&f64, &usize)> = m.iter()
                        .zip(m_i.iter())
                        .collect();
    zipped_m.sort_by(|a, b| a.0.partial_cmp(b.0).expect("Could not sort using m_1.4"));

    l_i = zipped_l.into_iter().map(|(_, i)| *i).collect();
    m_i = zipped_m.into_iter().map(|(_, i)| *i).collect();
    
    m_i.append(&mut l_i);

    m_i
}