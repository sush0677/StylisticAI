// static/script.js

$(document).ready(function() {
    // Smooth scrolling for links
    $('a[href^="#"]').on('click', function(event) {
        var target = $(this.getAttribute('href'));
        if (target.length) {
            event.preventDefault();
            $('html, body').stop().animate({
                scrollTop: target.offset().top
            }, 1000);
        }
    });

    // Handle questionnaire submission
    $('#submitQuestionnaire').click(function() {
        let answers = [
            $('#question1').val(),
            $('#question2').val(),
            $('#question3').val()
        ];

        if (answers.every(answer => answer.trim() !== "")) {
            // Call to the server or model to get recommendation based on answers
            let recommendation = "Based on your mood and preferences, we recommend a chic blue outfit for a calm yet vibrant look.";
            $('#recommendationText').text(recommendation);
            $('#questionnaireResult').fadeIn();
        } else {
            alert("Please answer all questions before submitting.");
        }
    });
});
