#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:24:33 2019

Purpose: This is a script to create the dash application for Activity detection. 

@author: zeski
"""

import os
import pickle
import base64

import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html

def encode_image(image_file): 
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())




# importing necessary materials --- Graphs, tables, images, etc. 

## Graphs and Tables: 
os.chdir("Plots_For_Dash_App")
class_balances       = pickle.load(open("class_balances.pickle", "rb"))
rf_baseline_train    = pickle.load(open("baseline_training.pickle", 'rb'))
rf_baseline_test     = pickle.load(open("baseline_testing.pickle", 'rb'))
feat_sel_prior       = pickle.load(open("Feature_Importances_pre.pickle", "rb"))
feat_sel_after       = pickle.load(open("feature_importances_selected.pickle", 'rb'))
GB_sel_train         = pickle.load(open("Gradient_Booster_Training_selected.pickle" ,"rb"))
GB_sel_test          = pickle.load(open("Gradient_Booster_Testing_selected.pickle" ,"rb"))
corr_target_feat     = pickle.load(open("correlations_with_target_variable.pickle", "rb"))
explained_var_prin   = pickle.load(open("explained_variance_prin_comp.pickle", "rb"))
prin_comp_vis        = pickle.load(open("principle_component_visualization.pickle", "rb" ))
select_feat_vis      = pickle.load(open("selected_features_visualization.pickle", "rb"))
xgb_prin_train       = pickle.load(open("gradient_booster_princomp_training.pickle", "rb")) 
xgb_prin_test        = pickle.load(open("gradient_booster_princomp_testing.pickle" , "rb"))

os.chdir("..")


# Dash Application code: 


app = dash.Dash()

server = app.server


app.layout = html.Div(children = [
        
        
        # Title Division
        html.Div( id = "Title", children = [
                
                html.H1(
                        "Activity Recognition: Model Building Process", 
                        style = {
                                "textAlign" : "center", 
                                "color"     : "rgb(30,144,255)", 
                                "fontSize"  : 40
                                }
                        )
                
                ]), 
          
            
        # Introduction to the problem    
        html.Div(id = "Intro", children = [
                html.H2(
                        "Introduction To The Problem: ", 
                        style = {
                                 "paddingTop" : 20
                                }
                        ),
                
                html.Div([dcc.Markdown(
                        children = ["""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Utilizing data retrieved from the UCI machine learning repository, 
                        I built a Gradient Boosting model to classify human activity. The goal of this project was to explore common preprocessing techniques, 
                        feature selection and extraction, as well as various machine learning algorithms to build a classifier for human activity recognition. 
                        The project is divided into separate components: \n\n\n\n \t 1: Initial Exploratory Data Analysis \n\n  \t 2: Feature Selection \n\n \t 3: Feature Extraction
                        \n\n\n\n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Through each section, I train and assess different machine learning algorithms to explore how well they perform given different data formats. I utilize confusion matrices, 
                        and roc plots to assess how well each model performs. \n\n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The associated live visualization 
                        can be found here: \n\n https://activity-detection-ml2.herokuapp.com/
                        
                        
                        """])],
 
                        style = {
                                
                                "textAlign" : "left",
                                "paddingLeft" : 15, 
                                "paddingRight" : 15, 
                                "width" : "45%", 
                                "display":"inline-block", 
                                'paddingBottom' : 100
            
                                }), 
                html.Div([
                        html.Img(src = encode_image('Plots_For_Dash_App/TonyStark.jpeg')), 

                        
                        ], style = {
                                 "display":"inline-block" , 
                                 'paddingLeft' : 150,
                                 'width': '45%'
                                        
                                })

]), 



        ## Exploring classes and Baseline Models
        html.Div(id = 'Exploring Classes and Models',children = [
        ### Containing the header and explanatory text
        html.Div(id = "Part 1", children= [
                html.Hr(),
                html.H2("Exploring Classes, Normality and Building a Baseline Model: "),
                
                html.Div([dcc.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The first task was to perform common EDA tasks, such as examining normaility, scale, counts, etc. 
                                       Throughout the entire model building process, there will be EDA tasks performed, however there were some 
                                       necessary tasks to perform before choosing to build a model. As this is a classification problem, 
                                       I wanted to get an idea as to what the class balances were. Determining whether or not a model 
                                       has balanced or imbalanced target classes can affect the type of model to be chosen, type of feature engineering(Resampling) to be done,
                                       and metrics to evaluate the model by. I plotted the counts of classes found within the data to visualize this. I found that the data has balanced classes. 
                                      \n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Another crucial first step is to establish the data
                                      types of the supporting features. The data types for this data set were primarily numerical, with the only categorical variable
                                      being the target variable, which was encoded and transformed into a discrete range. Because of the sheer amount of features present in the data, 
                                      performing a visualization of each feature and examining for normality was not pratical. I performed a komologrov smirnov test for each variable to 
                                      account for normality. To my surprise, non of the features were from a normal distribution according to the test. \n\n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; I visualized the first 6 variables via histogram and plotted these against a histogram drawn from a 
                        random distribution using the numpy library. The histograms revealed quite a lot, the data is undoubtedly non-normal, and skewed as well. This indicates presence of outliers, later confirmed by a 
                        utilizing the IQR to test for data points outside the upper and lower fences of the individual features. I chose the IQR over z-score because of non normality. After careful consideration, I decided against removing
                        the outliers because of potential weight towards classification. Instead I decided to choose a family of models robust to outliers ... the Tree or Ensemble family. As a preliminary analysis, I built a baseline model, 
                        and scored how well it performed on the training and testing utilizing a confusion matrix below. It performed fairly well, albeit very slowly because of high dimensional feature space. 
                                      
                                       
                        
                        """)], style = {'width': '60%', 'display' : 'inline-block', 'paddingLeft': 15}),
                
                
                
                ### Class balances Graph
                html.Div([
                    
                        dcc.Graph( id = 'class_balances', 
                                  figure = {
                                          'data'   : class_balances['data'], 
                                          'layout' : class_balances['layout'] 
                                          
                                          }
                                
                                
                                )
                        
                        ], style = {
                                "paddingTop"   : 30,
                                "paddingBottom": 5,
                                "width"        : "52%",
                                'display' : 'inline-block'

                        }), 
                
                
                ### Histogram Image
                html.Div(children = [
                        
                        html.Img(src = encode_image("Plots_For_Dash_App/Histograms_of_few_variables.png"))
                        
                        
                        ], style = {
                                    "width"    : "46%",
                                    'display' : 'inline-block', 
                                    "paddingRight" : 35
                
                                    })
                
                
                
                ], style = {"paddingTop": 20}), 
                
                html.Div(children = [
                        
                        ### RF baseline training
                        html.Div([dcc.Graph(id = 'RF_pre_feature_selection_training', 
                                  
                                  figure = {
                                          'data'    : rf_baseline_train['data'], 
                                          'layout'  : rf_baseline_train['layout']
                                          
                                          }
                                
                                
                                )], style = 
                                        {
                                          "width" : "48%", 
                                          'display': 'inline-block'
                                                }                                
                        ), 
                        
                
                        ### RF baseline testing
                        html.Div([dcc.Graph(id = "RF_pre_feature_selection_testing", 
                                  figure = {
                                          'data'   : rf_baseline_test['data'],
                                          'layout' : rf_baseline_test['layout']
                                          }
                                
                                
                                
                                )],style = { 
                                            "width" : "48%", 
                                            'display': 'inline-block'}),
                

                        
                        ], style = {'paddingTop': 25}),
                        
                
                

]),        

        
                
        ## Part 2: Feature Selection 
        
        html.Div(id = "feature selection", children = [
                
                ### Dividing line
                html.Hr(), 
                
                ### Header
                html.H2("Feature Selection and Model Refinement"),
                
                ### Pragraphs explaining work
                html.Div([
                        dcc.Markdown(
                            """ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This step of the process involves feature selection to reduce the dimensions
                            of the dataset. Since my baseline model is an ensemble method, I have decided to use the embedded tree method for feature selection and work from there. 
                            This will select features that are of the most importance when making splits. The splits that provide the best separations will be 
                            the features with higher importances. Since the model has high correlations, and a high dimensional feature space with nearly every feature 
                            containing substantial outliers, I have decided to go with a gradient boosted model. While a random forest can handle high dimensional feature 
                            spaces, it is considerably slower in execution time than the sequential gradient boosted model. Furthermore, the random forest is a high variance 
                            model seeking to improve on overfitting via reducing the variance of the individual trees. Whereas the gradient boosted model starts 
                            with a high bias and seeks to reduce bias by introducing more variance to the system through increasing max depth of trees. 
                            \n\n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; With the outliers, and high dimensions, 
                            I am concerned about starting with an algorithm that instrisincally posseses high variance and is nartuall prone to overfitting despite scaling back with hyper parameters. This is 
                            part of the reason I have chose the intrinsic biased gradient booster, to reduce potential of overfitting with a high variance. Instead, we can introduce variation into the model 
                            via tuning the hyper parameters. The other reason is the execution speed of gradient boosters far out perform execution speed of random forests, especially with multi threading. 
                            The resulting graphs from the feature selection using a gradient boosted model 
                            can be shown below.
                            \n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As you can see from the left graph, a majority of the features present in the original dataset carry little to no importance in 
                            making splits to segregate the classes. I chose to be slightly more conservative in selecting features by choosing a cut off value of 0.005. This preserved
                            58 of the original features, which can be displayed on the right graph with their new calculated importances. You can see the update in accuracy examining the confusion matrices below. 
                            To further whittle down on the features present in the dataset and examine the possibility of performing feature extraction, I plotted the correlations between 
                            variables in the new dataset, and the correlations between the features and the target variable. For nearly all of the features contain high correlation with target variable with a near zero probability. 
                            This inidcates that with a large enough collection of data (> 500 samples), that the correlations are likely to be signicant and not due to random variations in the data(<0.05 critical region). However, it is interesting 
                            to note that while the variables have high correlations with the target, there is also high multi-colinearity between the features themselves. This allows for the chance of performing feature extraction, namely 
                            Principle Component Analysis to capture the correlations between the variables and extract components that will preserve the variation of the data while reducing the dimensions even further! The next section will 
                            deal with feature extraction and the visual exploration of the classes and the final model.
 
                            """
                                
                                )
                        
                        ],style = {'width': '60%', 'display' : 'inline-block', 'paddingLeft': 15}), 
               ### Feature Importances Prior 
               html.Div([
                       
                       dcc.Graph( id = 'feature_importances_prior', 
                                 
                                 figure = {
                                         
                                         'data'   : feat_sel_prior['data'], 
                                         'layout' : feat_sel_prior['layout']
                                         
                                         }
                               
                               
                               
                               )
                       
                       
                       ], style = {"width" : "48%", 
                                    'display': 'inline-block'}),      
                ### Feature importances After
                html.Div(children = [
                        
                        dcc.Graph(id = 'feature_importances_after', 
                                  
                                  figure = {
                                          
                                          'data'   : feat_sel_after['data'], 
                                          'layout' : feat_sel_after['layout']
                                          
                                          }
                                )
                        
                        ],style = {"width" : "48%", 
                                    'display': 'inline-block'}), 
                        
                        
                        
                 ### Correlations between features and target vairable       
                 html.Div([
                         
                         dcc.Graph( id = 'correlation_target_&_features', 
                                   figure = {
                                           'data'   : corr_target_feat['data'], 
                                           'layout' : corr_target_feat['layout']
                                           
                                           }
                                 
                                 
                                 )
                         
                         
                         
                         
                         ],style = {"width" : "48%", 
                                    'display': 'inline-block',
                                    'paddingBottom' : 50
                                    }) , 
                         
                         
                 ### Correlations between the features        
                 html.Div([
                         
                          html.Img(src = encode_image("Plots_For_Dash_App/correlations_between_variables_after.png"))
                         
                         ],style = {"width" : "48%", 
                                    'display': 'inline-block',
                                    'paddingBottom' : 50
                                    }),
                
                
                
                
                
                
                ### XGB selected Training         
                html.Div(children = [
                        
                        dcc.Graph( id = 'XGB_selected_training', 
                                  
                                  figure = {
                                          'data'    : GB_sel_train['data'], 
                                          'layout'  : GB_sel_train['layout']
                                          
                                          
                                          }
                                
                                )
                        
                        
                        
                        
                        
                        
                        ],style = {"width" : "45%", 
                                    'display': 'inline-block'}),
                
                ### XGB selected Testing         
                html.Div(children = [
                        
                        dcc.Graph(id = 'XGB_selected_testing', 
                                  
                                  figure = {
                                          'data'   : GB_sel_test['data'], 
                                          'layout' : GB_sel_test['layout']
                                          
                                          
                                          }

                                )
                        
                        
                        
                        
                        
                        ],style = {"width" : "48%", 
                                    'display': 'inline-block'}), 
                        


]),

                        
        ## Part 3: Feature Extraction                
        html.Div(id = 'feature extraction', children= [
                
                html.Hr(),
                
                html.H2("Feature Extraction and Final Model"),
                
                ## Explanatory Paragraph part3
                html.Div([
                            dcc.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After considering the selected features and the correlations
                                         between features, I explored feature extraction to reduce the dimensions even further, and visualize the seperability better. Below I have plots 
                                         of the seperability between classes with the top 3 selected features from the gradient boosted feature selection, and the principle components
                                         derived from the total 58 features. Examining the feature seperation plot, with only the top three features there is good separability between the 
                                         classes. However, if we examine the explained variance from the principle components plot, by performing PCA we can extract and use even less features to explain a significant proportion of
                                         the variance. Below these two plots, is a scatter plot of the seperations of the classes utilizing the top three principle components (70% of variance). The plot takes on similar shape
                                         to the feature selection scatterplot above. However, with the top three components, there is a greater degree of seperation between classes and even the data points themselves. This allows for a 
                                         better visual representation of the seperation of classes.\n\n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For my final model, I have chosen to implement a gradient 
                                         boosted algorithm with the top 15 principle components(91% of the variance). With this model, you can see that it performs slightly worse on testing than its previous build in the model 
                                         refinement section above. This is to be expected when performing PCA. While allowing you to shrink the dimensions of the data and preserving the variation among it, it cannot preserve 100%. This will 
                                         explain the minor loss in performance. The dimensions of the dataset have been reduced from 561 features down to a mere 15. This vastly improves execution time of the model, while also maintaining strong predictive 
                                         performance. 
                                         
                                         
                                         
                                         
                                         """),                        
                        
                        
                        
                        ], style = {'width': '60%', 'display' : 'inline-block', 'paddingLeft': 15}),
                                         
                ## Selected Feature visualization , top three features, seperation of classes                                         
                html.Div([
                         dcc.Graph(
                                 figure = {
                                         'data'    : select_feat_vis['data'], 
                                         'layout'  : select_feat_vis['layout']
                                         
                                         
                                         }
                                 
                                 
                                 )
                         
                         ], style = {"width" : "48%", 
                                    'display': 'inline-block'}), 
                 
                ## Principle Compnent analysis
                html.Div([
                        dcc.Graph(
                                figure = {
                                        'data'   : explained_var_prin['data'], 
                                        'layout' : explained_var_prin['layout'] 
                                        
                                        
                                        }
                                
                                
                                )
                        
                        
                        ], style = {"width" : "48%", 
                                    'display': 'inline-block'}),
                 
                ## Extracted Feature visualization
                html.Div([
                        dcc.Graph(
                                figure = {
                                        "data"   : prin_comp_vis['data'], 
                                        'layout' : prin_comp_vis['layout']
                                        
                                        }
                                
                                
                                )
                        
                        ],style = {'display': 'inline-block', 'paddingLeft' : 300}), 
               
               ## confusion matrix training 
               html.Div([
                       dcc.Graph(
                               figure = {
                                       'data'   : xgb_prin_train['data'], 
                                       'layout' : xgb_prin_train['layout']
                                       
                                       }
                               
                               )
                       
                       
                       ],style = {"width" : "48%", 
                                    'display': 'inline-block'}),                 
               ## confusion matrix testing 
               html.Div([
                       dcc.Graph(
                               figure = {
                                       'data'    : xgb_prin_test['data'], 
                                       'layout'  : xgb_prin_test['layout']
                                       
                                       }
                               
                               
                               )
                       
                       
                       ], style = {"width" : "48%", 
                                    'display': 'inline-block'})   
                
                
                
                ]), 
                            
                            
                
                
        ## Final Notes and Comparisons        
        html.Div([
                
                ##Header and Paragraph
                
                html.Div([
                        html.H2("Final Notes and Comparisons"), 
                        dcc.Markdown(""" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Confusion matrices are not the only way to visualize performance of your models. 
                               Below I have ROC Curve plots to better visualize how well the model has performed. The basic intuition behind an ROC plot is to plot 
                               the True Positive rate (The rate in which classifications were accurate) as a function of the false positive rate(the rate in which classifications were inaccurate). 
                               The median line in the middle of plot represents a model that predicts classes correctly 50% of the time. The goal is to build a model whose curve rests above this line, preferably closer to the top 
                               left of the graph. The farther the line of class prediction shifts towards the top left of the graph, the more true positive predictions were made over false positives. \n\n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The Area Under the Curve (AUC) provides 
                               a metric with which to score how well your model has performed on classifcation of a particular class. A model with perfect performance will have an AUC of 1.0, whereas a model which completely fails will have 0.0, or rather 0.5(the model performs worse than if you were to randomly guess). 
                               Given this, if we are to look at the ROC plots below, we can see that the model with the selected features does perform better than the model with the principle components. However this is marginal because of the fact that the AUC's are nearly identical. 
                               Given the choice between the two models, I would choose to implement the model that has a faster execution time over the model with a slightly better accuracy. 
                               
                        
                        """)
                        
                        ], style = {'width': '60%', 'display' : 'inline-block', 'paddingLeft': 15}), 
                        
                        
                ###ROC Plot selected features        
                html.Div([
                        html.Img(src = encode_image("Plots_For_Dash_App/ROC_Plots_selected.png"))
                        
                        ],style = {"width" : "48%", "display" : "inline-block"}),
                        
                        
                ### ROC Plot Principle Components        
                html.Div([
                        
                        html.Img(src = encode_image("Plots_For_Dash_App/ROC_Plots_Components.png"))
                        
                        
                        ],style = {"width" : "48%", "display" : "inline-block"}),
                
                
                
                
                ]),
                
        
        
        
        
        
        
        
        
        
  
        
])



if __name__ =="__main__":
    app.run_server(debug = True)
