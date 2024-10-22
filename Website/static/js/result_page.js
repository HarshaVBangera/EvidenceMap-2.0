$(".collapseHeader").click(function () {
    $collapseHeader = $(this);
    //getting the next element
    $collapseContent = $collapseHeader.next();
    //open up the content needed - toggle the slide- if visible, slide up, if not slidedown.
    $collapseContent.slideToggle(500, function () {
    });

});
