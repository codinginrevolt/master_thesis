import dash
from dash import html
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
import numpy as np

from gaussianprocess import GP
from kernels import Kernel
from eos import EosProperties
import sampling as sam
import prepare_ceft as pc
import prepare_pqcd as pp
import anal_helpers as anal


n_ceft, cs2_ceft, cs2_l, cs2_u = anal.get_ceft_cs2()
indices0 = np.unique(np.int64(np.geomspace(1, 70, 40))) #20
indices0 = np.concatenate(([0],indices0))
indices = np.arange(71, len(n_ceft), 12) #48
indices = np.concatenate((indices0,indices))
n_ceft = n_ceft[indices]
cs2_ceft = cs2_ceft[indices]
cs2_l = cs2_l[indices]
cs2_u = cs2_u[indices]

_, phi_ceft_avg, phi_ceft_lower, phi_ceft_upper, = anal.get_ceft_phi()
phi_ceft_avg = phi_ceft_avg[indices]
phi_ceft_lower = phi_ceft_lower[indices]
phi_ceft_upper = phi_ceft_upper[indices]

phi_ceft_sigma = pc.CI_to_sigma(
                    phi_ceft_upper-phi_ceft_lower,
                    75)

cs2_hat, X_hat, sigma_hat, l_hat, alpha_hat = sam.get_hype_samples()
n_pqcd, cs2_pqcd = pp.get_pqcd(X_hat, size=200)
cs2_pqcd_sigma = np.zeros_like(cs2_pqcd)

phi_pqcd = pc.get_phi(cs2_pqcd)
phi_pqcd_sigma = np.zeros_like(phi_pqcd)

n = np.concatenate((n_ceft, n_pqcd))
phi = np.concatenate((phi_ceft_avg, phi_pqcd))
phi_sigma = np.concatenate((phi_ceft_sigma, phi_pqcd_sigma))

n_test = np.linspace(n[0], 40, 400)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Kernels and their Hyperparameters for GPR"),
    html.Div([
        dcc.Dropdown(
        id='kernel_type',
        options=[
            {'label': 'Squared Exponential',
             'value' : 'SE'},
            {'label':'Matern 3/2',
             'value': 'Matern32'},
            {'label':'Matern 5/2',
             'value': 'Matern52'},
            {'label':'Gamma-Exponential',
             'value': 'GE'},
            {'label':'Rational Quadratic',
             'value': 'RQ'}
            ],
        value='SE',
        style={'width': '50%'}
        ),
        ], 
    style = {
        'display':'flex',
        'alignItems': 'center'
        }
    ),
    
    html.Div([
        html.Label(dcc.Markdown('**$\\sigma$**', mathjax=True)), 
        dcc.Slider(id='sigma', min=0.1, max=3, step=0.1, value=1),
        html.Label(dcc.Markdown('**l**')),
        dcc.Slider(id='l', min=0.1, max=3, step=0.1, value=1),
         html.Label(dcc.Markdown('**$\\gamma$** (only for Gamma exponential)', mathjax=True)), 
        dcc.Slider(
            id='gamma', 
            min=0.1, max=2, step=0.1, value=2,
            ),

        html.Label(dcc.Markdown('**$\\alpha$** (only for RBF)', mathjax=True)), 
        dcc.Slider(
            id='alpha', 
            min=0.1, max=10, step=0.1, value=1,
            marks={
                0.1: '0.1',
                2: '2',
                4: '4',
                6: '6',
                8: '8',
                10: '10'
                }
                )
    ]),

    dcc.Graph(id='gp_plot'),

    html.Div(
        id='cond_number',
    ),

    dcc.Graph(id='cs2_plot')
])



@app.callback(
    Output('gp_plot', 'figure'),
    Input('kernel_type', 'value'),
    Input('sigma', 'value'),
    Input('l', 'value'),
    Input('gamma', 'value',),
    Input('alpha', 'value')
)
def update_gp(kernel_type, sigma, l, gamma, alpha):
    # Build kernel
    if kernel_type == 'SE':
        kernel = Kernel('SE', sigma=sigma, l=l)
    if kernel_type == 'Matern32':
        kernel = Kernel('Matern32', sigma=sigma, l=l)
    if kernel_type == 'Matern52':
        kernel = Kernel('Matern52', sigma=sigma, l=l)
    if kernel_type == 'GE':
        kernel = Kernel('GE', sigma=sigma, l=l, gamma = gamma)
    if kernel_type == 'RQ':
        kernel = Kernel('RQ', sigma=sigma, l=l, alpha=alpha)

    # Fit GP
    gp = GP(kernel, cs2_hat)
    gp.fit(n, n_test, phi, phi_sigma, stabilise=True)
    mu, std = gp.posterior()
    mu = mu.flatten()
    std = std.flatten()
    samples = gp.posterior(n=5, sampling=True) 

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n.flatten(), y=phi, mode='markers', name='Train'))
    fig.add_trace(go.Scatter(x=n_test.flatten(), y=mu, mode='lines', name='Mean'))
    fig.add_trace(go.Scatter(
        x=np.concatenate([n_test.flatten(), n_test.flatten()[::-1]]),
        y=np.concatenate([mu + std, (mu - std)[::-1]]),
        fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
        showlegend=True, name='Uncertainty'
    ))
    for i in range(samples.shape[0]):
        fig.add_trace(go.Scatter(
            x=n_test.flatten(),
            y=samples[i].flatten(),
            mode='lines',
            line=dict(width=1, dash='solid'),
            name=f'Samples',
            opacity=1,
            showlegend=(i==0)  # Show only one legend entry for all samples, or set to True for all
        ))

    fig.update_layout(
        title="Draws from single GP",
        xaxis_title="Number density [n_sat]",
        yaxis_title="Phi",
    )
    return fig



""" @app.callback(
    Output('cs2_plot', 'figure'),
    Input('kernel_type', 'value'),
    Input('sigma', 'value'),
    Input('l', 'value'),
    Input('gamma', 'value',),
    Input('alpha', 'value')
)
def update_gp(kernel_type, sigma, l, gamma, alpha):
    if kernel_type == 'SE':
        kernel = Kernel('SE', sigma=sigma, l=l)
    if kernel_type == 'Matern32':
        kernel = Kernel('Matern32', sigma=sigma, l=l)
    if kernel_type == 'Matern52':
        kernel = Kernel('Matern52', sigma=sigma, l=l)
    if kernel_type == 'GE':
        kernel = Kernel('GE', sigma=sigma, l=l, gamma = gamma)
    if kernel_type == 'RQ':
        kernel = Kernel('RQ', sigma=sigma, l=l, alpha=alpha)

    _, _, _, e_ini, p_ini, mu_ini, _, _, _, _ = pc.make_conditioning_eos()
    gp = GP(kernel, cs2_hat)
    gp.fit(n, n_test, phi, phi_sigma, stabilise=True)
    mu, std = gp.posterior()
    mu = mu.flatten()
    std = std.flatten()
    samples = gp.posterior(n=20, sampling=True) 
    fig = go.Figure()

    for i in range(samples.shape[0]):
        eos = EosProperties(n_test, samples[i], epsi_0=e_ini, p_0=p_ini, mu_0=mu_ini)
        cs2_test = (eos.get_all())["cs2"]

        fig.add_trace(go.Scatter(
            x=n_test.flatten(),
            y=cs2_test.flatten(),
            mode='lines',
            line=dict(width=1, dash='solid', color = 'blue'),
            name=f'Samples',
            opacity=0.5,
            showlegend=(i==0),
        ))

    n_ceft1, _, cs2_l, cs2_u = anal.get_ceft_cs2()
    fig.add_trace(go.Scatter(
        x = list(n_ceft1) + list(n_ceft1)[::-1],
        y = list(cs2_u) + list(cs2_l)[::-1],
        fill = 'toself',
        fillcolor = 'rgba(255,165,0,0.5)',
        line = dict(color='rgba(255,255,255,0)'), 
        showlegend = True,
        name = "CEFT Band"
    ))
    fig.update_layout(
        title="Draws from single GP",
        xaxis_title="Number density [n_sat]",
        yaxis_title="Sound Speed Squared",
    )
    return fig """

@app.callback(
    Output('cs2_plot', 'figure'),
    Input('kernel_type', 'value'),
    Input('sigma', 'value'),
    Input('l', 'value'),
    Input('gamma', 'value',),
    Input('alpha', 'value')
)
def update_cs2(kernel_type, sigma, l, gamma, alpha):
    # Build kernel
    if kernel_type == 'SE':
        kernel = Kernel('SE', sigma=sigma, l=l)
    if kernel_type == 'Matern32':
        kernel = Kernel('Matern32', sigma=sigma, l=l)
    if kernel_type == 'Matern52':
        kernel = Kernel('Matern52', sigma=sigma, l=l)
    if kernel_type == 'GE':
        kernel = Kernel('GE', sigma=sigma, l=l, gamma = gamma)
    if kernel_type == 'RQ':
        kernel = Kernel('RQ', sigma=sigma, l=l, alpha=alpha)

    _, _, _, e_ini, p_ini, mu_ini, _, _, _, _ = pc.make_conditioning_eos()
    fig = go.Figure()

    for i in range(10):
        cs2_hat, X_hat, _, _, _ = sam.get_hype_samples()
        n_pqcd, cs2_pqcd = pp.get_pqcd(X_hat, size=200)

        phi_pqcd = pc.get_phi(cs2_pqcd)
        phi_pqcd_sigma = np.zeros_like(phi_pqcd)

        n = np.concatenate((n_ceft, n_pqcd))
        phi = np.concatenate((phi_ceft_avg, phi_pqcd))
        phi_sigma = np.concatenate((phi_ceft_sigma, phi_pqcd_sigma))
        gp = GP(kernel, cs2_hat)
        gp.fit(n, n_test, phi, phi_sigma, stabilise=True)

        sample = gp.posterior(n=1, sampling=True)
        eos = EosProperties(n_test, sample, epsi_0=e_ini, p_0=p_ini, mu_0=mu_ini)
        cs2_test = (eos.get_all())["cs2"]

        fig.add_trace(go.Scatter(
            x=n_test.flatten(),
            y=cs2_test.flatten(),
            mode='lines',
            line=dict(width=1, dash='solid', color = 'blue'),
            name=f'Samples',
            opacity=0.5,
            showlegend=(i==0),
        ))

    n_ceft1, _, cs2_l, cs2_u = anal.get_ceft_cs2()

    fig.add_trace(go.Scatter(
        x = list(n_ceft1) + list(n_ceft1)[::-1],
        y = list(cs2_u) + list(cs2_l)[::-1],
        fill = 'toself',
        fillcolor = 'rgba(255,165,0,0.5)',
        line = dict(color='rgba(255,255,255,0)'), 
        showlegend = True,
        name = "CEFT Band"
    ))
    
    fig.update_layout(
        title="Draws from different GPs",
        xaxis_title="Number density [n_sat]",
        yaxis_title="Sound Speed Squared",
    )

    return fig



if __name__ == '__main__':
    app.run(debug=True)