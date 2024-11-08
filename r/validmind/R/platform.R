#' Retrieve a validmind (vm) connection object using reticulate
#'
#' @param api_key The ValidMind API key
#' @param api_secret The ValidMind API secret
#' @param model The ValidMind model
#' @param python_version The Python Version to use
#' @param api_host The ValidMind host, defaulting to local
#'
#' @importFrom reticulate import use_python py_config
#'
#' @return A validmind connection object, obtained from `reticulate`,
#' which orchestrates the connection to the ValidMind API
#'
#' @examples
#'\dontrun{
#' vm_r <- vm(
#'    api_key="<your_api_key_here>",
#'    api_secret="<your_api_secret_here>",
#'    model="<your_model_id_here>",
#'    python_version=python_version,
#'    api_host="https://api.dev.vm.validmind.ai/api/v1/tracking"
#'  )
#'}
#'
#' @export
vm <- function(api_key, api_secret, model, python_version,
               api_host = "http://localhost:3000/api/v1/tracking") {
  use_python(python_version)

  vm <- import("validmind")

  vm$init(
    api_host = api_host,
    api_key = api_key,
    api_secret = api_secret,
    model = model
  )

  return(vm)
}

#' Print a summary table of the ValidMind results
#'
#' @param result_summary A summary of the results
#'
#' @return A data frame containing the summary of the ValidMind results
#'
#' @importFrom glue glue
print_summary_tables <- function(result_summary) {
  return(result_summary$serialize(as_df = TRUE))
}

#' Provide a summarization of a single metric result
#'
#' @return A list containing the summary of the ValidMind results
#'
#' @param result The ValidMind result object
summarize_metric_result <- function(result) {
  if (result$result_id == "dataset_description") {
    return(NULL)
  }

  metric <- result$metric

  return(metric$summary)
}

#' Provide a summarization of a single test result
#'
#' @return A list containing the summary of the ValidMind test results
#'
#' @param result The ValidMind result object
summarize_test_result <- function(result) {
  if (result$result_id == "dataset_description") {
    return(NULL)
  }

  metric <- result$test_results

  return(metric$summary)
}

#' Provide a summarization of a single result (test or metric)
#'
#' @return Based on the type of `result`, either A list containing the summary
#' of the ValidMind results, or a list containing the summary of the ValidMind
#' results
#'
#' @param result The ValidMind result object
summarize_result <- function(result) {
  result_class <- class(result)[[1]]

  if (isTRUE(grepl("TestSuiteDatasetResult", result_class))) {
    # Ignore for now
    # print("TestPlanDatasetResult")
  } else if (isTRUE(grepl("TestSuiteMetricResult", result_class))) {
    summarize_metric_result(result)
  } else if (isTRUE(grepl("TestSuiteTestResult", result_class))) {
    summarize_test_result(result)
  }
}

#' Build an R Plotly figure from a JSON representation
#'
#' @return An R Plotly object derived from the JSON representation
#'
#' @param plotly_figure A nested list containing plotly elements
#'
#' @importFrom plotly plotly_build
build_r_plotly <- function(plotly_figure) {
  # Grab the plotly code as a list
  fig_list <- plotly_figure$figure$to_dict()

  # Extract data and layout
  p <- plotly_build(fig_list)

  # Return the R plotly plot
  return(p)
}

#' Process a set of ValidMind results into parseable data
#'
#' @param results A list of ValidMind result objects
#'
#' @importFrom dplyr bind_rows
#'
#' @return A nested list of ValidMind results (dataframes, plotly plots, and
#' matplotlib plots)
#' @export
#'
#' @examples
#'\dontrun{
#' vm_dataset = vm_r$init_dataset(
#'   dataset=data,
#'   target_column="Exited",
#'   class_labels=list("0" = "Did not exit", "1" = "Exited")
#' )
#'
#' tabular_suite_results <- vm_r$run_test_suite("tabular_dataset", dataset=vm_dataset)
#'
#' processed_results <- process_result(tabular_suite_results)
#' processed_results
#' }
#'
process_result <- function(results) {
  overall_result <- list()

  # Sequentially process every result in the result set
  for (index in 1:length(results$sections)) {
    # Grab the test suite
    suite <- results$sections[[index]]
    overall_result[[suite$section_id]] <- list()

    # Grab the individual test
    # print(glue("Test Suite Results: {suite$section_id}\n"))

    # Get path to temporary directory
    tmp_dir <- tempdir()

    # Process every result in that particular suite
    for (full_result in suite$tests) {
      if (!("title" %in% names(full_result))) full_result$title <- full_result$name
      # print(full_result$title)

      overall_result[[suite$section_id]][[full_result$title]] <- list()

      # Store a list of the possible results we will display
      plotly_images <- list()
      matplotlib_images <- list()
      result_tables <- list()

      result <- full_result$result
      description <- result$result_metadata[[1]]$text

      # Summarize the tables
      if ("metric" %in% names(result)) {
        if (!is.null(result$metric$summary)) {
          table_res <- result$metric$summary$results
          for (tbl in table_res) {
            try(
              {
                result_tables[[length(result_tables) + 1]] <- bind_rows(tbl$data)
              },
              silent = TRUE
            )
          }
        }
      }

      # Process and bind together all the summarized tabular results
      if ("test_results" %in% names(result)) {
        try(
          {
            table_res <- result$test_results$results

            full_table <- list()
            for (res in table_res) {
              my_tbl <- try(
                {
                  bind_rows(c(list("Column" = res$column), res$values))
                },
                silent = TRUE
              )

              if (inherits(my_tbl, "try-error")) {
                my_tbl <- bind_rows(res$values)
              }

              full_table[[length(full_table) + 1]] <- my_tbl
            }

            full_table <- bind_rows(full_table)

            if (!is.null(full_table) && nrow(full_table) > 0) {
              result_tables[[length(result_tables) + 1]] <- full_table
            }
          },
          silent = TRUE
        )
      }

      # Check if we actually have figures to process
      if ("figures" %in% names(result)) {
        # Process each figure one by one
        for (figure in result$figures) {
          # First check if it's a plotly figure
          if (figure$is_plotly_figure()) {
            plotly_images[[length(plotly_images) + 1]] <- build_r_plotly(figure)
            # Otherwise, it's a matplotlib figure
          } else if (figure$is_matplotlib_figure()) {
            # Original name
            orig_name <- figure$metadata$`_name`
            full_path <- file.path(tmp_dir, paste0(orig_name, ".png"))

            # Store if we haven't yet
            if (!(full_path %in% unlist(matplotlib_images))) {
              figure$figure$savefig(full_path)

              matplotlib_images[[length(matplotlib_images) + 1]] <- full_path
            }
          }
        }
      }

      final_result <- list(
        description = description,
        plotly_images = plotly_images,
        matplotlib_images = matplotlib_images,
        result_tables = result_tables
      )

      overall_result[[suite$section_id]][[full_result$title]] <- final_result
    }
  }

  return(overall_result)
}

#' Produce RMarkdown-compatible output of all results
#'
#' @param processed_results A list of processed result objects
#'
#' @importFrom dplyr %>%
#' @importFrom base64enc dataURI
#' @importFrom htmltools div HTML tags
#' @importFrom DT datatable
#'
#' @return A formatted list of RMarkdown widgets
#' @export
#'
#' @examples
#'\dontrun{
#' vm_dataset = vm_r$init_dataset(
#'   dataset=data,
#'   target_column="Exited",
#'   class_labels=list("0" = "Did not exit", "1" = "Exited")
#' )
#'
#' tabular_suite_results <- vm_r$run_test_suite("tabular_dataset", dataset=vm_dataset)
#'
#' processed_results <- process_result(tabular_suite_results)
#' all_widgets <- display_report(processed_results)
#' for (widget in all_widgets) {
#'   print(widget)
#' }
#'}
#'
display_report <- function(processed_results) {
  all_widgets <- list()

  for (section in names(processed_results)) {
    test_suites <- processed_results[[section]]
    for (suite in names(test_suites)) {
      # Create a temporary file for the markdown content
      temp_markdown_file <- tempfile(fileext = ".md")
      orig_text <- processed_results[[section]][[suite]]$description
      text_to_write <- glue(paste0("### {suite}\n\n", orig_text), "\n\n")
      widget_list <- list()

      if (is.character(orig_text)) {
        # Write the markdown string to the temporary file
        writeLines(text_to_write, temp_markdown_file)

        # Convert markdown to HTML
        temp_html_file <- tempfile(fileext = ".html")
        rmarkdown::pandoc_convert(
          input = temp_markdown_file,
          to = "html",
          output = temp_html_file
        )

        # Read the HTML content
        html_content <- readLines(temp_html_file, warn = FALSE)
        html_content <- paste(html_content, collapse = "\n")

        # Create a single widget
        widget_list <- list(description = div(style = "color: black;", HTML(html_content)))
      }

      for (t1 in processed_results[[section]][[suite]]$result_tables) {
        widget_list[[length(widget_list) + 1]] <- datatable(t1)
      }

      for (p in processed_results[[section]][[suite]]$plotly_images) {
        res <- try(p, silent = TRUE)

        if (!inherits(res, "try-error")) {
          widget_list[[length(widget_list) + 1]] <- p
        }
      }

      res <- unlist(processed_results[[section]][[suite]]$matplotlib_images)

      if (!is.null(res)) {
        for (im in res) {
          img_data <- dataURI(file = im, mime = "image/png")

          img_tag <- tags$img(
            src = img_data,
            alt = "Description of image",
            width = "100%", height = "auto"
          )

          widget_list[[length(widget_list) + 1]] <- img_tag
        }
      }

      if (length(widget_list) > 0) {
        combined_widget <- do.call(htmltools::tagList, widget_list)
        class(combined_widget) <- c(suite, section, "shiny.tag.list")

        # print(combined_widget)
        all_widgets[[length(all_widgets) + 1]] <- combined_widget
      }
    }
  }

  return(all_widgets)
}

#' Save an R model to a temporary file
#'
#' This function saves a given R model object to a randomly named `.RData` file
#' in the `/tmp/` directory. The file is saved with a unique name generated
#' using random letters.
#'
#' @param model The R model object to be saved.
#'
#' @return A string representing the full file path to the saved `.RData` file.
#'
#' @examples
#' model <- lm(mpg ~ cyl, data = mtcars)
#' file_path <- save_model(model)
#'
#' @export
save_model <- function(model) {
  random_name <- paste(sample(letters, 10, replace = TRUE), collapse = "")
  file_path <- file.path(tempdir(), paste0(random_name, ".RData"))
  save(model, file = file_path)

  return(file_path)
}
