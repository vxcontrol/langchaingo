package agents

import (
	"testing"

	"github.com/vxcontrol/langchaingo/schema"

	"github.com/stretchr/testify/require"
)

func TestMRKLOutputParser(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		input           string
		expectedActions []schema.AgentAction
		expectedFinish  *schema.AgentFinish
		expectedErr     error
	}{
		{
			input: "Action:  foo Action Input: bar",
			expectedActions: []schema.AgentAction{{
				Tool:      "foo",
				ToolInput: "bar",
				Log:       "Action:  foo Action Input: bar",
			}},
			expectedFinish: nil,
			expectedErr:    nil,
		},
		{
			input: "Action: foo\nAction Input:\nbar\nbaz",
			expectedActions: []schema.AgentAction{{
				Tool:      "foo",
				ToolInput: "bar\nbaz",
				Log:       "Action: foo\nAction Input:\nbar\nbaz",
			}},
			expectedFinish: nil,
			expectedErr:    nil,
		},
		{
			input: "Action: calculator\nAction Input: 5 + 3\nObservation:",
			expectedActions: []schema.AgentAction{{
				Tool:      "calculator",
				ToolInput: "5 + 3\nObservation:",
				Log:       "Action: calculator\nAction Input: 5 + 3\nObservation:",
			}},
			expectedFinish: nil,
			expectedErr:    nil,
		},
	}

	a := OneShotZeroAgent{}
	for _, tc := range testCases {
		actions, finish, err := a.parseOutput(tc.input)
		require.ErrorIs(t, tc.expectedErr, err)
		require.Equal(t, tc.expectedActions, actions)
		require.Equal(t, tc.expectedFinish, finish)
	}
}
