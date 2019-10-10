using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;

namespace ML
{
    //This class representing input data
    class FeedBackTraningData
    {
        [ColumnName("Label")]         public bool IsGood { get; set; }
        [ColumnName("FeedbackText")]          
        public string FeedBackText { get; set; }
    }
    
    class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }


    class Program
    {
        static List<FeedBackTraningData> trainingData = new List<FeedBackTraningData>();
        static List<FeedBackTraningData> testData = new List<FeedBackTraningData>();
        static void LoadTrainingData()
        {
            //Positive 001
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "This is good.",
                    IsGood = true
                }
            );
            //Nagetive 002
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "This is horrible.",
                    IsGood = false
                }
            );
            //Postive 003
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Very Positive",
                    IsGood = true
                }
            );
            //Nagetive 004
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "It seems that there is something wrong.",
                    IsGood = false
                }
            );
            //Nagetive 005
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Fuck you",
                    IsGood = false
                }
            );
            //Positive 006
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "You did incrdible excellent at your exercise.",
                    IsGood = true
                }
            );
            //Nagetive 007
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "The refugees were a pathetic sight ",
                    IsGood = false
                }
            );
            //Positive 008
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "You can make it.",
                    IsGood = true
                }
            );
            //Positive 009
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Follow your heart.",
                    IsGood = true
                }
            );
            //Postive 010
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "This is the most inspiring thing I've ever heard.",
                    IsGood = true
                }
            );
            //Nagetive 011
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Today a dark-skinned stranger will defame you .",
                    IsGood = false
                }
            );
            //Nagetive 012
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "You did your assigment horrible.",
                    IsGood = false
                }
            );
            //Nagetive 013
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "What a horrible day.",
                    IsGood = false
                }
            );
            //Positive 014
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Trust your guts.",
                    IsGood = true
                }
            );
            //Nagetive 015
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "This weather depresses me.",
                    IsGood = false
                }
            );
            //Nagetive 016
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "It depresses me to think that I'll probably still be doing exactly the same job in ten years' time.",
                    IsGood = false
                }
            );
            //Positive 017
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Good job. dude",
                    IsGood = true
                }
            );
            //Positive 018
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Have a good time.",
                    IsGood = true
                }
            );
            //Nagetive 019
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Got damn it",
                    IsGood = false
                }
            );     
            //Positive 020
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "You are so beautiful.",
                    IsGood = true
                }
            ); 
            //Positive 021
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Greet.",
                    IsGood = true
                }
            ); 
            //Positive 022
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Handsome.",
                    IsGood = true
                }
            );  
            //Nagetive 023
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "The assigment is horrible.",
                    IsGood = false
                }
            );     
            //Nagetive 023
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Horrible",
                    IsGood = false
                }
            ); 
            //Nagetive 023
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "His heart is ugly.",
                    IsGood = false
                }
            ); 
            //Nagetive 023
            trainingData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "I enjoy fucking around.",
                    IsGood = false
                }
            ); 
        }
        static void LoadTestData()
        {
            testData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "good",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "horrible",
                    IsGood = false
                }
            );

            testData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "What a horrible day.",
                    IsGood = false
                }
            );

            testData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "Got damn it",
                    IsGood = false
                }
            );   
            
            testData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "The assigment is horrible.",
                    IsGood = false
                }
            ); 
            testData.Add(
                new FeedBackTraningData(){
                    FeedBackText = "This weather depresses me.",
                    IsGood = false
                }
            );
        }
 
        static void Main(string[] args)
        {
            // Step1: Load Traning data
            LoadTrainingData();
            // Step2: Create object of MLContext
            var mlContext = new MLContext();
            // Step3: Convert input data into IdataView
            IDataView dataView = mlContext.Data.LoadFromEnumerable<FeedBackTraningData>(trainingData);
            // Step4: Create the pipe line and define the workflow in it. 
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "FeedbackText")
                           .Append(mlContext.BinaryClassification.Trainers.FastTree(
                               numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1
                           ));
            // Step5: Training the algorithm and we want the model out
            var model = pipeline.Fit(dataView);

            // Step6: Load the test data and run the test data to check our models accuracy
            LoadTestData();
            IDataView dataView1 = mlContext.Data.LoadFromEnumerable<FeedBackTraningData>(testData);
            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions,"Label");

            //testing 
            Console.WriteLine("Accuracy: " + metrics.Accuracy);

            // Step7: use the model
            string feedbackString;
            do
            {
                Console.WriteLine("Enter the feedback string:(Enter stop to stop)");
                feedbackString = Console.ReadLine();
                var predictionFunction = mlContext.Model.CreatePredictionEngine
                                                        <FeedBackTraningData,FeedBackPrediction>
                                                        (model);
                var feedbackInput = new FeedBackTraningData();
                feedbackInput.FeedBackText = feedbackString;
                var feedBackPredicted = predictionFunction.Predict(feedbackInput);
                if(feedbackString != "stop")
                {
                    if(feedBackPredicted.IsGood)
                    {
                        Console.WriteLine("Positive");
                    }
                    else
                    {
                        Console.WriteLine("Nagetive");
                    }
                }
            }while(feedbackString != "stop");
            


        } 
    }
}