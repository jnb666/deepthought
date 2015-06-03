import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.1
import GoExtensions 1.0

ApplicationWindow {
	id: root; title: "deepthought"; color: "lightgray"; 
	x: 50; y: 50
	minimumWidth: 800; minimumHeight: 900
	onClosing: ctrl.send("quit", "")

	TabView{
		id: tab		
		objectName: "tabs"
		anchors.left: parent.left; anchors.right: parent.right
		height: 450
		Component.onCompleted: {
			addTab("train", tab1)
			addTab("options", tab2)
		}

		Component {
			id: tab1
			ColumnLayout {	
				RowLayout {
					spacing: 20
					Button { text: "restart"; onClicked: ctrl.send("start", "") }
					Button { text: "step"; onClicked: ctrl.send("step", "") }
					Button {
						objectName: "runButton"; text: "run"; checkable: true
						onClicked: ctrl.send("run", checked ? "start" : "stop");
					}
					Button { text: "stats"; onClicked: ctrl.send("stats", "") }
					Label { text: "data set:" }
					ComboBox {
						objectName: "modelList"
						width: 120
						model: ListModel {
							{{ range .Datasets }}
							ListElement { text: "{{ . }}" }
							{{ end }}
						}
						currentIndex: {{ .Index }}
						onActivated: ctrl.select(index)
					}
				}
				Plots{
					id: plot; objectName: "plotControl"
					width: 800; height: 800
					background: "black"; color: "white"
					grid: true; gridColor: "#404040"
					Timer {
						interval: 100; running: true; repeat: true
						onTriggered: plot.update()
					}
				}
				RowLayout {
					ExclusiveGroup { id: plotGroup }
					{{ range .Plots }}
					RadioButton { 
						text: "{{ .Name }}"; exclusiveGroup: plotGroup; checked: {{ .Selected }}
						onClicked: plot.select({{ .Index }})
					}
					{{ end }}
				}
			}
		}

		Component {
			id: "tab2"
			GridLayout {
				onVisibleChanged: cfg.update()
				columns: 3
				columnSpacing: 5; rowSpacing: 5
				Label { 
					text: "number of runs"
					anchors.right: runs.left; anchors.rightMargin: 10
				}
				TextField {
					id: runs; objectName: "MaxRuns"
					validator: IntValidator{}
				}
				Button { 
					text: "save" 
				}
				Label { 
					text: "max epoch"
					anchors.right: epochs.left; anchors.rightMargin: 10
				}
				TextField {
					id: epochs; objectName: "MaxEpoch"
					validator: IntValidator{}
				}
				Button { 
					text: "load" 
				}
				Label { 
					text: "learning rate"
					anchors.right: eta.left; anchors.rightMargin: 10
				}
				TextField {
					id: eta; objectName: "LearnRate"
					validator: DoubleValidator{}
				}
				Label { 
					Layout.rowSpan: 8
				}
				Label {
					text: "weight decay"
					anchors.right: lambda.left;	anchors.rightMargin: 10
				}
				TextField { 
					id: lambda;	objectName: "WeightDecay"
					validator: DoubleValidator{}
				}
				Label {
					text: "momentum"
					anchors.right: mom.left; anchors.rightMargin: 10
				}
				TextField { 
					id: mom; objectName: "Momentum"
					validator: DoubleValidator{}
				}
				Label {
					text: "threshold"
					anchors.right: threshold.left; anchors.rightMargin: 10
				}
				TextField { 
					id: threshold; objectName: "Threshold"
					validator: DoubleValidator{}
				}
				Label {
					text: "batch size"
					anchors.right: batch.left; anchors.rightMargin: 10
				}
				TextField { 
					id: batch; objectName: "BatchSize"
					validator: IntValidator{}
				}
				Label {
					text: "stop after"
					anchors.right: stop.left; anchors.rightMargin: 10
				}
				TextField { 
					id: stop; objectName: "StopAfter"
					validator: IntValidator{}
				}
				Label {
					text: "log every"
					anchors.right: log.left; anchors.rightMargin: 10
				}
				TextField { 
					id: log; objectName: "LogEvery"
					validator: IntValidator{}
				}
				Label {
					text: "sampler"
					anchors.right: sampler.left; anchors.rightMargin: 10
				}
				ComboBox { 
					id: sampler; objectName: "Sampler"
					model: ["uniform", "random"]
				}					
			}
		}
	}
}