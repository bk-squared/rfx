A drawn PEC sheet on a traced graded mesh now follows the nodes resolved on
the nominal mesh when its normal axis is concrete. This applies to zero-thickness
PEC Boxes and PEC `add_thin_conductor` sheets. The in-plane extent moves with
those nodes, like `add_pinned_sheet`. A traced normal axis remains refused,
with the axis named in the message.
