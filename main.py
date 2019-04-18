import base64
import io
import dash
import os
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from gurobipy import *

project_list = None
df_choices = None
df_projects = None
n = None
p = None
s1 = None
s2 = None
s3 = None

COUT_CHOIX_1 = 0
COUT_CHOIX_2 = 1
COUT_CHOIX_3 = 5
COUT_CHOIX_4 = 20
COUT_NON_CHOISI = 1000

def parse_content(content):
    _, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_excel(io.BytesIO(decoded))

path = os.path.join(os.getcwd(), 'results')
if not os.path.exists(path):
    os.makedirs(path)

app = dash.Dash(__name__)
app.css.append_css({"external_url": [
    "static/style.css"
]})

app.layout = html.Div([
    html.Div([
        html.H1("Assignments", style={'margin-left':'10px', 'color':'#1F77B4'}),
        html.H4('Requirements', style={'margin-left':'10px', 'color':'#3BA071'}),
        html.Div([
            html.P('- Windows'),
            html.P('- Anaconda'),
            dcc.Markdown('\- A [gurobi licence](https://www.gurobi.com/documentation/8.1/quickstart_mac/retrieving_a_free_academic.html#subsection:academiclicense)')],
            style={'margin-left':'30px'}),
        html.H4('Upload here the project excel and the students choices excel', style={'margin-left':'10px', 'margin-top':'30px', 'color':'#3BA071'}),
        dcc.Upload(
            id='upload-projects',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '98%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin-right': 'auto',
                'margin-left':'auto',
                'margin-top' : '20px'
            },
            multiple=True
        ),
        html.H4('Add some informations about the projects', style={'margin':'30px 0px 10px 10px', 'color':'#3BA071'}),
        html.Table(id='projects-checkboxes')
    ], className="six columns"),
    html.Div([
        html.H4('Select a project to see the students choices', style={'color':'#3BA071'}),
        dcc.Dropdown(id='projects-dropdown', value=''),
        dcc.Graph(id='histogram', figure={
                                          'data': [{'x': ['1st choice', '2nd choice', '3rd choice', '4th choice'], 'y': [0, 0, 0, 0],
                                          'type': 'bar', 'name': 'Distribution'}]
                                         }),
        html.H4('Choose the weights to compute the cost matrix', style={'margin-bottom':'20px', 'color':'#3BA071'}),
        html.P('Cost of 1st choice', style={'float':'left', 'lineHeight': '35px', 'margin':'0px 20px 0px 10px', 'width':'250px', 'font-weight':'bold', 'color':'#1F77B4'}),
        dcc.Input(id='cout-choix-1', type='number', value=COUT_CHOIX_1), html.Br(), html.Br(),
        html.P('Cost of 2nd choice', style={'float':'left', 'lineHeight': '35px', 'margin':'0px 20px 0px 10px', 'width':'250px', 'font-weight':'bold', 'color':'#1F77B4'}),
        dcc.Input(id='cout-choix-2', type='number', value=COUT_CHOIX_2), html.Br(), html.Br(),
        html.P('Cost of 3rd choice', style={'float':'left', 'lineHeight': '35px', 'margin':'0px 20px 0px 10px', 'width':'250px', 'font-weight':'bold', 'color':'#1F77B4'}),
        dcc.Input(id='cout-choix-3', type='number', value=COUT_CHOIX_3), html.Br(), html.Br(),
        html.P('Cost of 4th choice', style={'float':'left', 'lineHeight': '35px', 'margin':'0px 20px 0px 10px', 'width':'250px', 'font-weight':'bold', 'color':'#1F77B4'}),
        dcc.Input(id='cout-choix-4', type='number', value=COUT_CHOIX_4), html.Br(), html.Br(),
        html.P('Cost of non choosen assignment', style={'float':'left', 'lineHeight': '35px', 'margin':'0px 20px 0px 10px', 'width':'250px', 'font-weight':'bold', 'color':'#1F77B4'}),
        dcc.Input(id='cout-choix-5', type='number', value=COUT_NON_CHOISI), html.Br(), html.Br(),
        html.Button('Compute', id='go-button', style={'margin-left':'auto', 'margin-right':'auto', 'display':'block', 'width':'100%', 'color':'#3BA071', 'background-color':'#EEEEEE', 'margin-top':'20px'}),
        html.Br(),
        html.Div([html.P('Results are in the folder :', style={'float':'left', 'margin-right':'10px'}),
                  html.P(str(path), style={'text-decoration':'underline'})]),
        html.H4('Distribution of the assignments', style={'margin-top':'20px', 'margin-bottom':'10px', 'color':'#3BA071'}),
        html.P('No output yet', id='error-out'),
        dcc.Graph(id='hist_affe', figure={
                                         'data': [
                                                 {'x': ['1st choice', '2nd choice', '3rd choice', '4th choice', 'Other affectation'],
                                                  'y': [0, 0, 0, 0, 0], 'type': 'bar', 'name': 'Affectations'}
                                        ]})
    ], className="six columns")
], className="row")

@app.callback([Output('projects-checkboxes', 'children'),
               Output('projects-dropdown', 'options'),],
              [Input('upload-projects', 'contents')])
def projects_dropdown_checkboxes(contents):
    if contents is not None:
    
        global project_list, df_choices, df_projects, n, p, s1, s2, s3
        
        for content in contents:
            df = parse_content(content)
            if df.columns[2] == 'NOM':
                df_choices = df
            elif df.columns[0] == 'Titre du projet':
                df_projects = df
        
        if df_choices is not None and df_projects is not None:
            
            project_list = list(df_projects['Code du projet'].dropna())
            project_names = list(df_projects['Titre du projet'].dropna())
            
            df_choices = df_choices.iloc[:,[2,3,4,5,6,7,8,9,10]]
            df_choices.columns = ['Nom', 'Prenom', 'Filiere', 'Libre', 'Depart', '1', '2', '3', '4']
            df_choices = df_choices[df_choices['Libre'] == 'NON']
            df_choices = df_choices[~df_choices['Depart'].str.contains('complète')]
            df_choices.reset_index(inplace=True, drop=True)
            n = len(df_choices)
            p = len(project_list)
            s1 = np.array(df_choices['Depart'].str.contains('1'))
            s2 = np.array(df_choices['Depart'].str.contains('2'))
            s3 = np.array(df_choices['Depart'].str.contains('ESIEE'))
            
            options = [{'label':project_name, 'value':project} for project, project_name in zip(project_list, project_names)]
            
            table = [html.Tr([html.Th(col, style={'color':'#1F77B4'}) for col in ['Force open', 'Project', 'Minimum of all year student needed',
                                                       'Maximum of student on the project each semester']])] + \
                    [html.Tr([
                        html.Td(dcc.Checklist(id='0'+str(i),options=[{'label':'','value':project_list[i]}], values=[])),
                        html.Td(project_names[i]),
                        html.Td(dcc.Input(id='1'+str(i),type='number', value=4)),
                        html.Td(dcc.Input(id='2'+str(i),type='number', value=6))
                    ]) for i in range(p)]
            
            return table, list(options)
    
    return [], []

@app.callback(Output('histogram', 'figure'),
             [Input('projects-dropdown', 'value')])
def draw_hist(project):
    if df_choices is not None:
        y = np.zeros(4)
        for choices in df_choices.values[:,-4:]:
            for i in range(4):
                if choices[i] == project:
                    y[i] += 1
        figure={'data': [{'x': ['1st choice', '2nd choice', '3rd choice', '4th choice'], 'y': y, 'type': 'bar', 'name': 'Distribution'}]}
        return figure
    return {'data': [{'x': ['1st choice', '2nd choice', '3rd choice', '4th choice'], 'y': [0, 0, 0, 0], 'type': 'bar', 'name': 'Distribution'}]}

@app.callback([Output('hist_affe', 'figure'),
               Output('error-out', 'children'),
               Output('error-out', 'style')],
              [Input('go-button', 'n_clicks'),
               Input('projects-checkboxes', 'children')])
def compute(n_clicks, table):
    if df_choices is not None and df_projects is not None and n_clicks is not None:
        
        project_arg = {project:arg for arg, project in enumerate(project_list)}
        cost = [COUT_CHOIX_1, COUT_CHOIX_2, COUT_CHOIX_3, COUT_CHOIX_4]
        c = np.full((n,p), COUT_NON_CHOISI)
        for i in range(n):
            for j in range(4):
                c[i, project_arg[df_choices.iloc[i, j-4]]] = cost[j]
        
        force_open = np.zeros(p, dtype='uint8')
        min_project = np.zeros(p, dtype='uint8')
        max_project = np.zeros(p, dtype='uint8')
        
        for i in range(p):
            line = table[i + 1]['props']['children']
            if len(line[0]['props']['children']['props']['values']) > 0:
                force_open[i] = 1
            min_project[i] = line[2]['props']['children']['props']['value']
            max_project[i] = line[3]['props']['children']['props']['value']
        
        model = Model('Student Assignment')
        
        a = model.addVars(n, p, vtype=GRB.BINARY, name="affectations")
        O = model.addVars(p, vtype=GRB.BINARY, name="ouvertures")
        model.update()
        
        obj = quicksum(a[(i,j)] * c[i,j] for i,j in [(k,l) for k in range(n) for l in range(p)])
        model.setObjective(obj, GRB.MINIMIZE)
        
        model.addConstrs((quicksum(a[(i,j)] for j in range(p)) == 1 for i in range(n)), "c1")
        model.addConstrs((a[(i,j)] <= O[j] for i,j in [(k,l) for k in range(n) for l in range(p)]), "c2")
        model.addConstrs((quicksum(a[(i,j)] * s3[i] for i in range(n)) + (1 - O[j]) * min_project[j] >= min_project[j] for j in range(p)), "c3")
        model.addConstrs((quicksum(a[(i,j)] * (s3[i] + s1[i]) for i in range(n)) <= max_project[j] for j in range(p)), "c4")
        model.addConstrs((quicksum(a[(i,j)] * (s3[i] + s2[i]) for i in range(n)) <= max_project[j] for j in range(p)), "c5")
        for j, v in enumerate(force_open):
            if v == 1:
                model.addConstr(O[j] == 1, "c6_" + str(j))
        model.update()
        
        try:
        
            model.optimize()
            
            affectations = np.zeros((n,p), dtype='uint8')
            for (i, j), v in a.items():
                affectations[i, j] = np.abs(v.x)
            affectations = np.array(project_list)[np.argwhere(affectations == 1)[:,1]]
            
            y = np.zeros(5, dtype='int32')
            for i, choices in enumerate(df_choices.values[:,-4:]):
                for j, choice in enumerate(choices):
                    if affectations[i] == choice:
                        y[j] += 1
                        break
            y[4] = n - y.sum()
            
            data = np.concatenate((df_choices['Nom'].values.reshape(n, 1),
                                   df_choices['Prenom'].values.reshape(n, 1),
                                   affectations.reshape(n, 1)), axis=1)
            
            df = pd.DataFrame(data, columns=['Nom', 'Prénom', 'Projet'])
            df.sort_values(['Nom', 'Prénom'], ascending=[1, 1], inplace=True)
            df.to_excel(os.path.join(path, 'results.xlsx'), index=False)
        
        except:
            return {'data': [
                            {'x': ['1st choice', '2nd choice', '3rd choice', '4th choice', 'Other affectation'],
                             'y': [0, 0, 0, 0, 0], 'type': 'bar', 'name': 'Affectations'}
                            ]}, 'This is not feasable (' + str(n_clicks) + ')', {'color':'#C13636'}
        
        return {'data': [
                        {'x': ['1st choice', '2nd choice', '3rd choice', '4th choice', 'Other affectation'],
                         'y': y, 'type': 'bar', 'name': 'Affectations'}
                        ]}, 'Solution found (' + str(n_clicks) + ')', {'color':'#3BA071'}
    
    return {'data': [
                    {'x': ['1st choice', '2nd choice', '3rd choice', '4th choice', 'Other affectation'],
                     'y': [0, 0, 0, 0, 0], 'type': 'bar', 'name': 'Affectations'}
                    ]}, 'No output yet', None

@app.callback(Output('cout-choix-1', 'step'),
             [Input('cout-choix-1', 'value')])
def define_choix_1(value):
    global COUT_CHOIX_1
    COUT_CHOIX_1 = value
    return None

@app.callback(Output('cout-choix-2', 'step'),
             [Input('cout-choix-2', 'value')])
def define_choix_2(value):
    global COUT_CHOIX_2
    COUT_CHOIX_2 = value
    return None

@app.callback(Output('cout-choix-3', 'step'),
             [Input('cout-choix-3', 'value')])
def define_choix_3(value):
    global COUT_CHOIX_3
    COUT_CHOIX_3 = value
    return None

@app.callback(Output('cout-choix-4', 'step'),
             [Input('cout-choix-4', 'value')])
def define_choix_4(value):
    global COUT_CHOIX_4
    COUT_CHOIX_4 = value
    return None

@app.callback(Output('cout-choix-5', 'step'),
             [Input('cout-choix-5', 'value')])
def define_choix_5(value):
    global COUT_NON_CHOISI
    COUT_NON_CHOISI = value
    return None


if __name__ == '__main__':
    app.run_server(debug=True)