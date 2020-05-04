library(shiny)
library(shinydashboard)
library(shinyjs)
library(shinythemes)
library(ggplot2)
library(scales)
library(shinyFiles)
library(DT)

library(randomForest)
library(gtools)
############################### ui ###################################
ui <- dashboardPage(
  skin = "purple",
  dashboardHeader(title = "NTHU Project Team"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dataset", tabName = "dataset", icon = icon("table")),
      menuItem("Parameters", tabName = "parameters", icon = icon("list-alt"))
    )
  ), # sidebar end
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "dataset",
              fluidRow(
                box(title = "Input Data(.csv)",height = 685, width = 3, status = "primary",
                    fileInput("file4", "Data1"),
                    fileInput("file5", "Data2")),
                
                box(title = "Data Table", status = "primary", width = 9, solidHeader = T, 
                    div(DT::dataTableOutput("table1"))
                )
              ) # fluidRow end
      ), # tab1 end
      
      # Second tab content
      tabItem(tabName = "parameters",
              fluidRow(
                box(title = "1. Product Condition 1", height = 200, width = 4, status = "info",
                    uiOutput("W_D")),
                box(title = "2. Product Condition 2", height = 200, width = 4, status = "info",
                    uiOutput("Vendor"),
                    uiOutput("Size")),
                box(title = "3. Product Condition 3", height = 200, width = 4, status = "info",
                    uiOutput("Cap")),
                box(title = "Parameter Tuning", status = "info", width = 12, solidHeader = T,
                    actionButton("start", "Start"),actionButton("reset", "Reset"),br(),
                    useShinyjs(),
                    div(shinyjs::useShinyjs(), id = "form",
                        uiOutput("para_tune"))),
                box(title = "Values", status = "info", width = 4, solidHeader = T,
                    tableOutput("values")),
                box(title = "Result", status = "warning", width = 8, solidHeader = T,
                    "2nd Bond Time(ms):",br(),
                    verbatimTextOutput("txtout"),
                    plotOutput("barplot"),
                    "Alarm Rate Prediction:",br(),
                    "TYPE1",br(),
                    verbatimTextOutput("alarmrate_SHTL"),
                    "TYPE2",br(),
                    verbatimTextOutput("alarmrate_NSOL"),
                    "TYPE3",br(),
                    verbatimTextOutput("alarmrate_EFO"))
              ) # fluidRow end
      ) # tab2 end
    ) # tabs end
  ) # body end
) # page end

############################### Run APP #######################################
shinyApp(ui, server)
